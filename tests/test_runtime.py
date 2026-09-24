"""Access boundaries and replay bookkeeping, using synthetic software fixtures."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from gui_joint_control.runtime import Slot,Prediction,run_trajectory,save_replay
from gui_joint_control.privacy import PrivacyLedger
from gui_joint_control.features import regional_rgb,public_projection
from gui_joint_control.cli import manifest_rows,candidate_seed,main,load_config


class RuntimeTests(unittest.TestCase):
    def test_shipped_reference_config_matches_controller_contract(self):
        config=load_config(Path(__file__).resolve().parents[1]/'configs/naacl_reference.json')
        self.assertFalse(config['reproduces_reported_results'])
        self.assertEqual(config['reference_controller']['observation_dim'],28)

    def test_decoder_streams_do_not_depend_on_other_counts_or_call_order(self):
        small=[candidate_seed(2026,'episode',12,i) for i in range(3)]
        _=[candidate_seed(2026,'earlier',0,i) for i in range(20)]
        large=[candidate_seed(2026,'episode',12,i) for i in range(20)]
        self.assertEqual(small,large[:3])
        self.assertEqual(len(set(large)),20)
        self.assertNotEqual(small[0],candidate_seed(2026,'episode',13,0))

    def test_projection_cli_fits_and_saves_new_numeric_artifact(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);records=base/'fixture.npz';output=base/'fit'
            rng=np.random.default_rng(13)
            np.savez(records,features=rng.normal(size=(1,25,256)),native_target_tokens=rng.normal(size=(1,25,16)))
            main(['fit-projection','--records',str(records),'--output',str(output)])
            matrix=np.load(output/'projection.npy',allow_pickle=False)
            self.assertEqual(matrix.shape,(16,256))
            self.assertTrue(np.isfinite(matrix).all())
            self.assertTrue(json.loads((output/'losses.json').read_text()))
            with self.assertRaises(FileExistsError):
                main(['fit-projection','--records',str(records),'--output',str(output)])

    def test_exhaustion_keeps_denominator_and_adds_discounted_miss_tail(self):
        slots=[Slot('click',(.4,.4,.6,.6)) for _ in range(8)]
        reads=[];observations=[]
        def load(t):reads.append(t);return np.zeros((25,256))
        def allocate(obs):observations.append(obs);return np.full(25,5.),1
        def executor(release,text,k):
            self.assertEqual(release.shape,(25,256));self.assertEqual(text,'click')
            return Prediction((.5,.5),.25)
        records,replay=run_trajectory(slots,load,allocate,executor)
        self.assertEqual(len(records),56)
        self.assertEqual(sum(r['eligible'] for r in records),8)
        self.assertEqual(reads,list(range(6)))
        self.assertEqual(sum(r['invoked'] for r in records),6)
        self.assertEqual(records[6]['status'],'FILTER_EXHAUSTED')
        self.assertIsNone(records[6]['executed_budgets'])
        self.assertEqual(len(records[0]['executed_budgets']),25)
        self.assertEqual(len(records[0]['refinement_scales']),25)
        self.assertAlmostEqual(replay[-1]['reward'],1.5-.99-.99**2)
        self.assertTrue(replay[-1]['terminal'])
        self.assertTrue(np.all(replay[-1]['next_observation']==0))
        self.assertAlmostEqual(float(observations[1][25]),.25)

    def test_mismatched_masks_rejected_before_screen_access(self):
        slots=[Slot('click',(.4,.4,.6,.6)),Slot('',None,False,True)]
        ledger=PrivacyLedger([False,True])
        with self.assertRaisesRegex(ValueError,'exact fixed'):
            run_trajectory(slots,lambda t:self.fail('read'),lambda o:None,lambda *a:None,ledger=ledger)

    def test_save_is_non_pickle_and_refuses_existing_output(self):
        _,records=run_trajectory([Slot('click',(.4,.4,.6,.6))],lambda t:np.zeros((25,256)),
            lambda o:(np.full(25,1.5),1),lambda *args:Prediction((.5,.5),0))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'replay.npz';save_replay(records,path)
            with np.load(path,allow_pickle=False) as data:
                self.assertEqual(data['observation'].shape,(1,28));self.assertEqual(data['success'].dtype,np.bool_)
            with self.assertRaises(FileExistsError):save_replay(records,path)

    def test_replay_preserves_public_population_bindings_without_pickle(self):
        _, records=run_trajectory([Slot('click',(.4,.4,.6,.6))],lambda t:np.zeros((25,256)),
            lambda o:(np.full(25,1.5),1),lambda *args:Prediction((.5,.5),0))
        records[0].update(slot_id='["trajectory",0]', next_slot_id='',
                          source_manifest_sha256='a'*64, task='G', family_id='family-1')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'replay.npz';save_replay(records,path)
            with np.load(path,allow_pickle=False) as data:
                for key in ('slot_id','next_slot_id','source_manifest_sha256','task','family_id'):
                    self.assertEqual(data[key].dtype.kind,'U')
                    self.assertEqual(data[key][0],records[0][key])

    def test_inference_helper_delegates_joint_and_standalone_with_explicit_ids(self):
        import torch
        from types import SimpleNamespace
        from gui_joint_control.runtime import inference_allocator
        calls=[]
        schedule=object()
        def proposal(state, *, slot_ids, tms_schedule):
            calls.append((state.shape,slot_ids,tms_schedule))
            return {'budgets':torch.full((1,25),2.), 'candidate_count':torch.tensor([7])}
        trainer=SimpleNamespace(device='cpu', proposal_action=proposal)
        joint=inference_allocator(trainer)
        budgets,count=joint(np.zeros(28))
        self.assertEqual(count,7);self.assertTrue(np.all(budgets==2))
        self.assertEqual(calls[-1],(torch.Size([1,28]),None,None))
        standalone=inference_allocator(trainer,slot_id_provider=lambda observation:'public-slot',tms_schedule=schedule)
        standalone(np.zeros(28))
        self.assertEqual(calls[-1],(torch.Size([1,28]),['public-slot'],schedule))
        with self.assertRaisesRegex(ValueError,'nonempty public ID'):
            inference_allocator(trainer,slot_id_provider=lambda observation:None)(np.zeros(28))

    def test_projection_orthogonal_and_regions_have_expected_shape(self):
        projection=public_projection(seed=1,encoder_dim=256)
        np.testing.assert_allclose(projection@projection.T,np.eye(256),atol=1e-12)
        patches=regional_rgb(Image.new('RGB',(11,17),(255,0,0)))
        self.assertEqual(patches.shape,(25,3,224,224))
        self.assertTrue(np.isfinite(patches).all())

    def test_manifest_rejects_duplicate_ids_and_wrong_task(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'manifest.jsonl'
            path.write_text('{"trajectory_id":"a","slot":0}\n'*2)
            with self.assertRaisesRegex(ValueError,'Duplicate'):manifest_rows(path)
            path.write_text('{"trajectory_id":"a","slot":0,"task":"A"}\n')
            with self.assertRaisesRegex(ValueError,'Grounding'):manifest_rows(path)


if __name__=='__main__':unittest.main()
