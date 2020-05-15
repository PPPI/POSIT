def convert_to_iterator_for_transformer(list_iter, cache=False, with_lid=True, max_length=24):
    import tensorflow as tf
    from tensorflow.data import Dataset

    out_types = (tf.float32, tf.float16, tf.float16) if with_lid else (tf.float32, tf.float16)
    dataset = Dataset.from_generator(lambda: list_iter, output_types=out_types)
    dataset = dataset.filter(lambda *x: tf.size(x[0]) < max_length)

    if cache:
        dataset.cache()
    return dataset