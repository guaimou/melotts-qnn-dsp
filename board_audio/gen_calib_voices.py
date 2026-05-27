"""Generate 30 calib samples each for a2 and a13."""
import numpy as np, os, sys, glob
sys.path.insert(0, '/home/fibo/AI model/tts_models')
sys.path.insert(0, '/home/fibo/AI model/tts_models/before/melo_minimal/melo_minimal_bundle/runtime')
from tts import _qnn_encode

texts = [
    '你好，欢迎使用语音合成系统。','今天天气真不错。','人工智能技术正在改变世界。',
    '请帮我查询明天的天气预报。','这个项目进展顺利。','我喜欢在周末听音乐看书。',
    '深度学习需要大量训练数据。','语音合成技术取得了很大进步。','祝你生日快乐健康幸福。',
    '现在是下午三点二十五分。','请按照说明书步骤操作。','这部电影评分很高值得看。',
    '自动驾驶需要处理复杂路况。','中国航天事业取得瞩目成就。','保护环境是每个人的责任。',
    '数学是自然科学的基础学科。','这家餐厅的菜品味道不错。','请确认订单信息是否正确。',
    '冬天北方会下很大的雪。','手机已成为不可或缺的工具。','量子计算有望解决传统难题。',
    '博物馆里展出许多珍贵文物。','植物通过光合作用转化二氧化碳。','这个故事让我想起童年。',
    '定期体检有助于发现健康问题。','新版本修复了之前的一些错误。','奥运会是世界盛大体育赛事。',
    '阅读可以开阔视野增长知识。','请在前方路口右转直行两百米。','科学研究需要严谨创新思维。',
]

for voice in ['a2', 'a13']:
    out_dir = '/home/fibo/melotts_qnn/calib_%s' % voice
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for i, text in enumerate(texts):
        try:
            z, T_enc = _qnn_encode(text, voice=voice, speed=1.0)
            if T_enc > 128:
                z = z[:, :, :128]
            elif T_enc < 128:
                pad = 128 - T_enc
                z = np.pad(z[0], ((0,0),(0,pad)), mode='constant')[np.newaxis,:,:]
            p = '%s/z_%04d.raw' % (out_dir, i)
            z.astype(np.float32).tofile(p)
            count += 1
        except Exception as e:
            print('  [%s][%d] FAIL: %s' % (voice, i, str(e)[:80]))

    with open('%s/calib_list.txt' % out_dir, 'w') as f:
        for i in range(len(texts)):
            p = '%s/z_%04d.raw' % (out_dir, i)
            if os.path.exists(p):
                f.write(p + '\n')

    files = sorted(glob.glob('%s/z_*.raw' % out_dir))
    if files:
        all_z = np.concatenate([np.fromfile(p, dtype=np.float32) for p in files])
        print('%s: %d files, z: min=%.4f max=%.4f mean=%.4f std=%.4f' % (
            voice, len(files), all_z.min(), all_z.max(), all_z.mean(), all_z.std()))

print('DONE')
