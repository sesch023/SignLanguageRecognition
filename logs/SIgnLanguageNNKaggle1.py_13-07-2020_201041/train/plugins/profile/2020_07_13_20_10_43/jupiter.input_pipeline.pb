  *	c;�Oy4�@2e
.Iterator::Model::BatchV2::Shuffle::Zip[0]::Mapd��Ljh�U@!�1�	�V@)d�� w�U@1Q4έ�U@:Preprocessing2e
.Iterator::Model::BatchV2::Shuffle::Zip[1]::Mapd]�mO��!@!M�����!@)�
G�J� @1�9^�ۭ @:Preprocessing2]
&Iterator::Model::BatchV2::Shuffle::Zipd����qX@!�����X@)7+1���?1H|�����?:Preprocessing2O
Iterator::Model::BatchV2e�����X@!���8e�X@)���b)�?1i?v�<P�?:Preprocessing2r
;Iterator::Model::BatchV2::Shuffle::Zip[1]::Map::TensorSliced�o�[��?!�y�i| �?)�o�[��?1�y�i| �?:Preprocessing2X
!Iterator::Model::BatchV2::Shuffled��r�X@!n����X@)�������?1�t��?:Preprocessing2r
;Iterator::Model::BatchV2::Shuffle::Zip[0]::Map::TensorSlicedu�׃I��?!���;&�?)u�׃I��?1���;&�?:Preprocessing2F
Iterator::ModelՒ�r0�X@!      Y@) �={.�?1�b �X�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.