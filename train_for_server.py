import tensorflow as tf

male_model = tf.keras.applications.InceptionV3(weights=None, include_top=True, input_shape=(224,224,1), classes=1, classifier_activation=None)
female_model = tf.keras.applications.InceptionV3(weights=None, include_top=True, input_shape=(224,224,1), classes=1, classifier_activation=None)
male_model.compile(loss='mae', optimizer='adam')
female_model.compile(loss='mae', optimizer='adam')
import pandas as pd

csv_train=pd.read_csv('./boneage-training-dataset.csv')

for i in range(len(csv_train['id'])):
  csv_train.loc[i,"id"] = f"{csv_train['id'][i]}.png"

print("\n******************** boneage-training-dataset ****************************\n")

male_df = csv_train.loc[csv_train.male == True]
female_df = csv_train.loc[csv_train.male == False]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # random으로 validation 뽑음

train_generator_m = datagen.flow_from_dataframe(male_df, directory='./boneage-training-dataset', x_col='id', y_col='boneage',
                                              target_size=(224,224), color_mode='grayscale',
                                              class_mode='raw', subset='training')
valid_generator_m = datagen.flow_from_dataframe(male_df, directory='./boneage-training-dataset', x_col='id', y_col='boneage',
                                              target_size=(224,224), color_mode='grayscale', 
                                              class_mode='raw', subset='validation', shuffle=False)


train_generator_f = datagen.flow_from_dataframe(female_df, directory='./boneage-training-dataset', x_col='id', y_col='boneage',
                                              target_size=(224,224), color_mode='grayscale',
                                              class_mode='raw', subset='training')
valid_generator_f = datagen.flow_from_dataframe(female_df, directory='./boneage-training-dataset', x_col='id', y_col='boneage',
                                              target_size=(224,224), color_mode='grayscale', 
                                              class_mode='raw', subset='validation', shuffle=False)


print("\n******************** MAKE Gegerator ****************************\n")


checkpoint_m = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath="./check_point_m")
checkpoint_f = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', save_best_only=True, filepath="./check_point_f")

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history_m = male_model.fit_generator(train_generator_m, validation_data=valid_generator_m, epochs=1, callbacks=[checkpoint_m, early_stopping])
history_f = female_model.fit_generator(train_generator_f, validation_data=valid_generator_f, epochs=1, callbacks=[checkpoint_f, early_stopping])

# male_model.fit(train_generator_m, steps_per_epoch=100, epochs=1, validation_data=valid_generator_m, validation_steps=800)
# female_model.fit(train_generator_f, steps_per_epoch=100, epochs=1, validation_data=valid_generator_f, validation_steps=800)

# male_model.save(./check_point_m)
# female_model.save(./check_point_f)
print("\n******************** FINISH save model ****************************\n")
