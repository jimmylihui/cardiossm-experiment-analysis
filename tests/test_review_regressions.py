"""Reproduce the two confirmed second-review regressions with synthetic arrays."""
import argparse
import copy
import csv
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cardiossm_training import data
from cardiossm_training.history import Pool
from cardiossm_training.options import TrainingChoices
from cardiossm_analysis.runs import partition_runs
from cardiossm_analysis.temporal import rolling_average, between_reading_segments, change_pairs
from cardiossm_analysis.scales import scale_metrics
from cardiossm_analysis.adapter import align_training_outputs
from cardiossm_analysis.calibration import assemble_cohort
from cardiossm_analysis import cli


class ReviewRegressionTests(unittest.TestCase):
    def threshold_recording(self):
        t=np.r_[np.arange(6)*20.,120+np.arange(6)*20.,240+np.arange(6)*10.,
                2100+np.arange(6)*10.,2160+np.arange(6)*10.]
        return dict(t=t,availability_s=t+10,map=np.repeat([0.,0.,0.,100.,100.],6),
            sig=np.zeros((len(t),3,1000),np.float32),fid=np.zeros((len(t),3,42),np.float32),
            hr_bpm=np.full(len(t),70.),retained_mask=np.ones(len(t),bool))

    def test_one_partition_reaches_history_adapter_hr_and_assembly(self):
        raw=self.threshold_recording()
        self.assertEqual(raw['availability_s'][18]-raw['availability_s'][17],1810.)
        for basis,context,level in [('window_completion',1,100.),('block_completion',19,25.)]:
            with self.subTest(basis=basis):
                d=data.retained(raw,'sample',run_gap_basis=basis)
                self.assertEqual(d['Tb'][3]-d['Tb'][2],1800.)
                pool=Pool({'sample':d},['sample']);_,mask=pool.contexts(pool.ids('sample',[18]))
                self.assertEqual(mask.sum(1).tolist(),[context]*3)
                train=data.smooth(d['Yb'],d['Tb'],d['rid'])
                analysis=rolling_average(d['Yb'],d['Tb'],run_id=d['rid'])
                self.assertEqual(train[3],level);np.testing.assert_array_equal(train,analysis)
                for i,g in enumerate(d['groups']):
                    np.testing.assert_array_equal(d['window_run_id'][g],np.full(len(g),d['rid'][i]))
                # Put the boundary in calibration too: reference/HR variance
                # must use the identical partition passed through the adapter.
                test=np.arange(5)>=4
                output=dict(Tb=d['Tb'],Yb=d['Yb'],te=test,raw_t=d['t'],raw_bid=d['bid'],ub=d['ub'],
                    availability_s=d['availability_s'],run_id=d['rid'],window_run_id=d['window_run_id'],
                    run_gap_basis=np.asarray(basis),HRb=d['HRb'],IP4_block=d['Yb'],IP4_positions=np.array([4]),
                    IP4_test_mask=test,IP4_calibration_mask=~test,IP4_readings=np.r_[np.full(4,np.nan),100.])
                record=align_training_outputs({'Signal':output},'IP4')
                self.assertAlmostEqual(float(record['calibration_variance']),0. if context==1 else 1054.6875)
                _,audit=assemble_cohort([record,copy.deepcopy(record)],['Signal'])
                self.assertEqual(audit['candidate_predictions'][0][3,1,0],level)
                bad=copy.deepcopy(output);bad['window_run_id'][18:]=9
                with self.assertRaises(ValueError):align_training_outputs({'Signal':bad},'IP4')

    def test_shared_ids_also_control_segments_pairs_and_scale_bins(self):
        t=np.arange(80)*120.;runs=np.r_[np.zeros(40,int),np.ones(40,int)]
        y=np.sin(np.arange(80)/3)*10;test=np.ones(80,bool);read=np.zeros(80,bool);read[0]=True
        segments=between_reading_segments(t,y,test,read,run_id=runs)
        self.assertEqual(len(segments),2)
        a,b=change_pairs(t,test,read,run_id=runs)
        self.assertTrue(np.all(runs[a]==runs[b]))
        result=scale_metrics(t,y,y,test,[9600],run_id=runs)
        self.assertEqual(result[0]['observations'],2)
        self.assertEqual(rolling_average(y,t,run_id=runs)[40],y[40])

    def test_missing_policy_and_blocks_straddling_window_runs_are_rejected(self):
        with self.assertRaises(ValueError):data.retained(self.threshold_recording(),'sample')
        choices=TrainingChoices(no_initial_test_scope='all')
        with self.assertRaises(ValueError):choices.validate('bcgnet',['pretraining_only'])
        with self.assertRaises(ValueError):partition_runs([10.,2000.],np.array([0,0]),[2000.],'window_completion')
        ids,_=partition_runs([10.,2000.],np.array([0,0]),[2000.],'block_completion')
        np.testing.assert_array_equal(ids,[0,0])

    def test_slow_csv_has_only_level_endpoints_and_comparison_guards_scales(self):
        t=np.arange(300)*120.;y=80+20*np.sin(np.arange(300)/5)
        read=np.zeros(300,bool);read[0]=True
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/'inputs';inputs.mkdir()
            np.savez_compressed(inputs/'sample.npz',time_s=t,reference=y,test_mask=np.ones(300,bool),
                reading_mask=read,run_id=np.zeros(300,int),run_gap_basis=np.asarray('window_completion'),
                model_names=np.array(['Signal']),neural_predictions=y[:,None])
            output={}
            for scale in ('block','slow'):
                args=argparse.Namespace(input=str(inputs),output=str(root/(scale+'.csv')),assemble_offset_hr=False,
                    assemble_models=None,network_only=True,network_scale=scale,qa=None,candidate_output=None,
                    concordance_no_reading=False)
                cli.score(args)
                with Path(args.output).open() as f:output[scale]=list(csv.DictReader(f))[0]
            self.assertGreater(int(output['block']['drop_positives']),0)
            self.assertTrue(any(k.startswith('rise_') for k in output['block']))
            self.assertFalse(any(k.startswith(('drop_','rise_')) or k in ('concordance','change_pairs') for k in output['slow']))
            self.assertEqual(output['slow']['reference_scale'],'past_only_60min')
            self.assertEqual(float(output['slow']['mae']),0.)
            compare=argparse.Namespace(input=str(root/'slow.csv'),output=str(root/'comparison.csv'),reference='Signal',
                metric='drop_auroc',aggregation='median',contrast='difference_of_aggregates',tuning='fixed',resamples=20,seed=0)
            with self.assertRaisesRegex(ValueError,'slow target'):cli.compare(compare)
            cli.write_csv(root/'mixed.csv',[output['block'],output['slow']]);compare.input=str(root/'mixed.csv');compare.metric='mae'
            with self.assertRaisesRegex(ValueError,'matching reference_scale'):cli.compare(compare)
            # Legacy run-free input must never silently choose the block clock.
            with np.load(inputs/'sample.npz') as z:old={k:z[k] for k in z.files if k not in ('run_id','run_gap_basis')}
            np.savez_compressed(inputs/'sample.npz',**old)
            with self.assertRaisesRegex(ValueError,'run_id'):cli.score(args)


if __name__=='__main__':unittest.main()
