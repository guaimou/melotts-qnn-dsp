import copy
import importlib

from . import cleaned_text_to_sequence


language_module_names = {
    "ZH": "chinese",
    "JP": "japanese",
    "EN": "english",
    "ZH_MIX_EN": "chinese",
    "KR": "korean",
    "FR": "french",
    "SP": "spanish",
    "ES": "spanish",
}

_loaded_language_modules = {}


def get_language_module(language):
    module_name = language_module_names[language]
    if module_name not in _loaded_language_modules:
        _loaded_language_modules[module_name] = importlib.import_module(
            f"{__package__}.{module_name}"
        )
    return _loaded_language_modules[module_name]


def clean_text(text, language):
    language_module = get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


def clean_text_bert(text, language, device=None):
    language_module = get_language_module(language)
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)

    word2ph_bak = copy.deepcopy(word2ph)
    for i in range(len(word2ph)):
        word2ph[i] = word2ph[i] * 2
    word2ph[0] += 1
    bert = language_module.get_bert_feature(norm_text, word2ph, device=device)

    return norm_text, phones, tones, word2ph_bak, bert


def text_to_sequence(text, language):
    norm_text, phones, tones, word2ph = clean_text(text, language)
    return cleaned_text_to_sequence(phones, tones, language)


if __name__ == "__main__":
    pass
