import tensorflow as tf
model = tf.keras.models.load_model('best_model.hdf5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( 'model.tflite' , 'wb' ) 
file.write( tflmodel )