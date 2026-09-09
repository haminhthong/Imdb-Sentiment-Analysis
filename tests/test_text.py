from sentiment.text import PAD_TOKEN, UNK_TOKEN, build_vocabulary, encode_and_pad, tokenize


def test_tokenize_giu_phu_dinh_va_loai_html():
    assert tokenize("<br>I don't like THIS!</br>") == ["i", "don't", "like", "this"]


def test_vocabulary_co_token_dac_biet_va_loc_tan_suat():
    vocabulary = build_vocabulary(["good movie", "good story"], min_frequency=2)
    assert vocabulary.token_to_index[PAD_TOKEN] == 0
    assert vocabulary.token_to_index[UNK_TOKEN] == 1
    assert "good" in vocabulary.token_to_index
    assert "movie" not in vocabulary.token_to_index


def test_encode_and_pad_tra_dung_do_dai_that():
    vocabulary = build_vocabulary(["a good movie"], min_frequency=1)
    encoded, length = encode_and_pad("good unknown", vocabulary, max_length=4)
    assert length == 2
    assert len(encoded) == 4
    assert encoded[-1] == vocabulary.pad_index
