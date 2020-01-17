from flask import *
import ast
import scipy.io.wavfile as wav
import numpy as np
import speechpy
import os
from firebase import firebase
import pyrebase
import urllib3
import io
from base64 import b64decode



firebase=firebase.FirebaseApplication("https://npci-database.firebaseio.com/",None)





app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def basic():
    varified=0
    autoid=''
    if request.method=="POST":
        req_data=request.get_json()
        print(req_data['recording'])
        if(req_data['number']=='3'):
            last=1
        else:
            last=0
        if(req_data['work']=='train'):
            file_name = '/Recording6'+'.wav'
            b64=req_data['recording']
            bin=b64decode(b64)
            f=io.BytesIO(bin)
            
            fs, signal = wav.read(f)
            signal = signal[:,0]
            autoidd=req_data['register']
            if(autoidd=='new'):
                flag=0
            else:
                flag=1
                autoid=autoidd
            signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)


            frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
                    zero_padding=True)

            power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
            print('power spectrum shape=', power_spectrum.shape)

            mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
            mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
            print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

            mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
            print('mfcc feature cube shape=', mfcc_feature_cube.shape)

            logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
            logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
            print('logenergy features=', logenergy.shape)
            if(flag==0):
                data={
                'userId':str(req_data['userid']),
                'Recording'+str(req_data['number']):str(req_data['recording']),
                'Score'+str(req_data['number']):str(power_spectrum.shape),
                'Recording2':'',
                'Score2':'',
                'Recording3':'',
                'Score3':'',
                'Scoreavg':''
            }
                result=firebase.post('/',data)
                autoid=result['name']
                print(result['name'])
            else:
                firebase.put('/'+autoid,'Recording'+str(req_data['number']),str(req_data['recording']))
                firebase.put('/'+autoid,'Score'+str(req_data['number']),str(power_spectrum.shape))
                result='Updated'
        else:
            autoid=req_data['register']
            b64=req_data['recording']
            bin=b64decode(b64)
            f=io.BytesIO(bin)
            fs, signal = wav.read(f)
            signal = signal[:,0]

    
            signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)


            frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
                    zero_padding=True)

            power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
            print('power spectrum shape=', power_spectrum.shape)

            mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
            mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
            print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

            mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
            print('mfcc feature cube shape=', mfcc_feature_cube.shape)

            logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                        num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
            logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
            print('logenergy features=', logenergy.shape)

            resultt=firebase.get('/'+autoid,'')
            sav=resultt['Scoreavg']
            sava=float(sav)
            powst=str(power_spectrum.shape)
            powst=powst.split(",")[0]
            powst=powst[1:]
            diff=abs(sava-int(powst))
            print(diff)
            if(diff>50):
                varified=0
            else:
                varified=1
                        
    if(last==1):
        scoav=firebase.get('/'+autoid,'')
        s1=scoav['Score1']
        s2=scoav['Score2']
        s3=scoav['Score3']
        s1=s1.split(",")[0]
        s1=s1[1:]
        s2=s2.split(",")[0]
        s2=s2[1:]
        s3=s3.split(",")[0]
        s3=s3[1:]
        savage=(int(s1)+int(s2)+int(s3))/3
        firebase.put('/'+autoid,'Scoreavg',str(savage))


    return jsonify({'messege':'Success','autoId':autoid,'varified':str(varified)})

if __name__=='__main__':
    app.run(debug=True)
