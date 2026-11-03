"""Geometry, development isolation, architecture and legacy deployment regressions."""
import numpy as np
import pytest
import torch

from shared.pose_geometry import (GEOMETRY_DIM, GEOMETRY_NAMES, GEOMETRY_VERSION,
                                  geometry_features, validate_contract)
from shared.pose_features import FEATURE_VERSION
from shared.model_improvement import DevelopmentData, candidate_config, ensure_development
from shared.mlp_model import PostureMLP, build_classifier_model


def test_geometry_numpy_torch_parity_and_base_unchanged():
    x=np.random.default_rng(1).normal(size=(20,13,4)).astype(np.float32)
    x[:,:,2:]=1
    x[:,[3,9],:]=0
    flat=x.reshape(-1,52)
    result=geometry_features(flat)
    np.testing.assert_array_equal(result[:,:52],flat)
    np.testing.assert_allclose(result,geometry_features(torch.from_numpy(flat)).numpy(),atol=1e-6,rtol=1e-6)
    assert result.shape==(20,96) and len(GEOMETRY_NAMES)==22
    assert np.isfinite(result).all()
    assert np.isin(result[:,-22:],[0,1]).all()
    assert np.all(result[:,52:74][result[:,74:]==0]==0)


def test_geometry_missing_and_coincident_joints_are_safe():
    absent=np.zeros((2,52),np.float32)
    assert np.all(geometry_features(absent)==0)
    coincident=absent.reshape(-1,13,4).copy();coincident[:,:,2:]=1
    result=geometry_features(coincident.reshape(-1,52))
    assert np.isfinite(result).all()
    # Directions and angles with zero-length vectors are explicitly invalid.
    assert (result[:,74:][:,[0,1,6,7,8,9,14,15]]==0).all()


def test_geometry_is_frame_local_and_joint_masks_propagate():
    x=np.ones((3,52),np.float32)
    x.reshape(-1,13,4)[:,5,:]=0
    full=geometry_features(x)
    np.testing.assert_array_equal(full[:1],geometry_features(x[:1]))
    assert np.all(full[:,74:][:,[0,1,2,3,6,7,8,10,11,16,17,18]]==0)


def test_original_architecture_and_new_models_have_six_outputs():
    for dim in [52,GEOMETRY_DIM]:
        for arch in ['mlp_small','mlp_base','mlp_wide','mlp_residual','mlp_layernorm']:
            model=build_classifier_model(arch,.15,dim).eval()
            assert model(torch.zeros(2,dim)).shape==(2,6)
    assert isinstance(build_classifier_model('mlp_wide'),PostureMLP)


def test_test_subject_cannot_enter_development():
    ensure_development(['CHANH','VMTRI'])
    with pytest.raises(ValueError,match='prohibited'):
        ensure_development(['CHANH','QBAO'])


def test_quality_gate_does_not_augment_sparse_or_low_confidence_pose():
    x=torch.ones(3,13,4);x[:,:,:2]=.4;x[:,:,2]=.9
    x[0,5:,:]=0
    x[1,:,2]=.4
    data=object.__new__(DevelopmentData);data.input_dim=52
    cfg={**candidate_config('targeted_arms'),'jitter_prob':1,'occlusion_prob':1}
    got=data.training_features(x.reshape(-1,52),cfg,torch.Generator().manual_seed(3)).reshape(-1,13,4)
    torch.testing.assert_close(got[:2],x[:2],rtol=0,atol=0)
    assert (got[2,:,3]==0).sum()==1
    assert got[2,0,3]==1 and got[2,5,3]==1 and got[2,11,3]==1


@pytest.mark.parametrize('architecture',['mlp_residual','mlp_layernorm'])
def test_geometry_contract_export_loader_round_trip(tmp_path,architecture):
    pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    from shared.dataset_pipeline import COCO13
    from shared.real_training import write_json
    from scripts.export_onnx import export
    from inference.pipeline import PostureClassifier
    contract=dict(version=GEOMETRY_VERSION,flatten_dim=96,shape_per_frame=[13,4],
                  channels=['x_norm','y_norm','confidence','valid_mask'],joints=COCO13,
                  confidence_threshold=.35,ema_alpha=.25,geometry_names=GEOMETRY_NAMES,
                  geometry_layout='base52,values22,masks22')
    assert validate_contract(contract)==96
    model=build_classifier_model(architecture,.15,96)
    checkpoint=dict(stage='final',model_name=architecture,model_config={'dropout':.15},
                    feature_contract=contract,label_names=[str(i) for i in range(6)],
                    model_state_dict=model.state_dict())
    torch.save(checkpoint,tmp_path/'model_final.pt')
    write_json(tmp_path/'feature_contract.json',contract)
    export(tmp_path)
    pt=PostureClassifier(tmp_path,'pt');onnx=PostureClassifier(tmp_path,'onnx')
    x=np.ones((13,4),np.float32)
    np.testing.assert_allclose(pt.predict(x),onnx.predict(x),atol=1e-5,rtol=1e-5)
    assert pt.predict(x).shape==(6,)
    with pytest.raises(ValueError,match='contract'):
        pt.predict(np.ones(53,np.float32))
