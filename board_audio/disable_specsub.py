c = open('/home/fibo/AI model/tts_models/tts.py').read()
c = c.replace('QNN_SPECSUB_ENABLED = True', 'QNN_SPECSUB_ENABLED = False')
open('/home/fibo/AI model/tts_models/tts.py','w').write(c)
print('specsub disabled')
