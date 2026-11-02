"""Generated block arrays only; no study data or retained output artifacts."""
import argparse
import copy
import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cardiossm_analysis.adapter import align_training_outputs
from cardiossm_analysis.calibration import assemble_cohort, other_patient_reading_variance, offset_hr
from cardiossm_analysis.bootstrap import paired_interval, holm_adjust
from cardiossm_analysis import cli


class ProtocolRepairTests(unittest.TestCase):
    def records(self):
        t=np.arange(150)*120.;cal=np.arange(150)<30;read=np.zeros(150,bool);read[[30,80]]=True
        records=[]
        for i in range(4):
            y=80+i+6*np.sin(np.arange(150)/9)
            records.append(dict(subject='sample'+str(i),time_s=t,reference=y,test_mask=~cal,
                reading_mask=read,calibration_mask=cal,hr_bpm=70+2*np.cos(np.arange(150)/8),
                run_id=np.zeros(len(t),int),run_gap_basis=np.asarray('window_completion'),
                model_names=np.array(['Temporal','Window']),neural_predictions=np.column_stack([y-3,y+2])))
        return records

    def test_cohort_qa_excludes_own_test_outcomes_and_keeps_baseline_network(self):
        original=self.records();first,audit=assemble_cohort(original,['Temporal'])
        altered=copy.deepcopy(original);altered[0]['reference'][~altered[0]['calibration_mask']]+=40
        _,second=assemble_cohort(altered,['Temporal'])
        np.testing.assert_array_equal(audit['selected_indices'][0],second['selected_indices'][0])
        for r,source in zip(first,original):
            np.testing.assert_array_equal(r['predictions'][:,2],source['neural_predictions'][:,1])
        self.assertEqual(audit['candidate_errors'].shape,(4,3,4))

    def test_no_calibration_R_excludes_own_variance_and_predicts_before_noisy_reading(self):
        np.testing.assert_array_equal(other_patient_reading_variance([1,3,9]),[31,30,27])
        t=np.arange(4)*120.;y=np.full(4,80.);zeros=np.zeros(4);read=np.array([1,0,1,0],bool)
        first=offset_hr(t,y,y,zeros,zeros,np.zeros(4,bool),read,qa=2.,reading_values=np.array([90,np.nan,100,np.nan]),observation_variance=30)[1]
        second=offset_hr(t,y+500,y+500,zeros,zeros,np.zeros(4,bool),read,qa=2.,reading_values=np.array([90,np.nan,150,np.nan]),observation_variance=30)[1]
        np.testing.assert_array_equal(first[:3],second[:3]);self.assertGreater(second[3],first[3])
        self.assertEqual(first[0],0.)

    def test_explicit_statistical_estimands_and_holm_family(self):
        a=np.array([.1,.2,.9]);b=np.array([0.,.8,.8])
        aggregate=paired_interval(a,b,'auroc',resamples=50,aggregation='median')
        delta=paired_interval(a,b,'auroc',resamples=50,contrast='median_paired_delta')
        self.assertAlmostEqual(aggregate['difference'],-.6);self.assertAlmostEqual(delta['difference'],.1)
        mean=paired_interval(a,b,'r_test',resamples=50,aggregation='mean')
        self.assertAlmostEqual(mean['difference'],float(np.mean(a-b)))
        np.testing.assert_allclose(holm_adjust([.04,.01,.03]),[.06,.03,.06])

    def training_output(self):
        t=np.arange(6)*120.;test=np.arange(6)>=2;pos=np.array([2]);readings=np.full(6,np.nan);readings[pos]=82.
        return dict(Tb=t,Yb=np.arange(6)+80.,te=test,raw_t=np.arange(36)*20.,raw_bid=np.repeat(np.arange(6),6),
            ub=np.arange(6),availability_s=np.arange(36)*20.+10,HRb=np.full(6,70.),IP4_block=np.arange(6)+78.,
            run_id=np.zeros(6,int),window_run_id=np.zeros(36,int),run_gap_basis=np.asarray('window_completion'),
            IP4_positions=pos,IP4_test_mask=test,IP4_calibration_mask=~test,IP4_readings=readings)

    def test_adapter_checks_support_and_shared_readings_without_double_smoothing(self):
        first=self.training_output();second=copy.deepcopy(first);second['IP4_block']+=2
        out=align_training_outputs({'one':first,'two':second},'IP4')
        np.testing.assert_array_equal(out['neural_predictions'][:,0],first['IP4_block'])
        second['IP4_readings'][2]+=1
        with self.assertRaises(ValueError):align_training_outputs({'one':first,'two':second},'IP4')
        second=copy.deepcopy(first);second['availability_s']+=1
        with self.assertRaises(ValueError):align_training_outputs({'one':first,'two':second},'IP4')

    def test_score_and_reselection_cli_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);inputs=path/'inputs';inputs.mkdir()
            for i,r in enumerate(self.records()):
                np.savez_compressed(inputs/('sample'+str(i)+'.npz'),**{k:v for k,v in r.items() if k!='subject'})
            args=argparse.Namespace(input=str(inputs),output=str(path/'metrics.csv'),assemble_offset_hr=True,
                assemble_models=['Temporal'],network_only=False,network_scale='block',qa=None,
                candidate_output=str(path/'candidates.csv'),concordance_no_reading=False)
            cli.score(args)
            with (path/'metrics.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),12)
            self.assertTrue(all(r['qa_selection']=='other_patient_test_mae' for r in rows if r['model']=='Temporal'))
            comparison=argparse.Namespace(input=str(path/'candidates.csv'),output=str(path/'comparison.csv'),
                reference='Window',metric='mae',aggregation='mean',contrast='difference_of_aggregates',
                tuning='reselect',resamples=20,seed=0)
            cli.compare(comparison)
            with (path/'comparison.csv').open() as f:result=list(csv.DictReader(f))
            self.assertTrue(all(r['reselects_qa']=='True' for r in result))

    def test_prepare_reads_fold_directories_and_rejects_repository_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('one','two'):
                for fold in ('f0','f1'):
                    target=root/name/fold;target.mkdir(parents=True)
                    np.savez_compressed(target/(fold+'.npz'),**self.training_output())
            output=root/'aligned'
            cli.prepare(argparse.Namespace(models=['one='+str(root/'one'),'two='+str(root/'two')],tag='IP4',output=str(output)))
            self.assertEqual(sorted(p.stem for p in output.glob('*.npz')),['f0','f1'])
        repository=Path(cli.__file__).resolve().parents[2]
        with self.assertRaises(ValueError):cli.external_output(repository/'outputs/forbidden.csv')


if __name__=='__main__':unittest.main()
