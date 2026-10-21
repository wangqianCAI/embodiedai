import os
def install(train):
    original = train.make_pre_post_processors
    def offline_processors(*args, **kwargs):
        overrides = dict(kwargs.get("preprocessor_overrides") or {})
        tokenizer = dict(overrides.get("tokenizer_processor") or {})
        tokenizer["tokenizer_name"] = os.path.join(os.environ.get("SO101_ROOT", "/data/so101_pi0"), "models/paligemma_tokenizer")
        overrides["tokenizer_processor"] = tokenizer
        kwargs["preprocessor_overrides"] = overrides
        return original(*args, **kwargs)
    train.make_pre_post_processors = offline_processors
