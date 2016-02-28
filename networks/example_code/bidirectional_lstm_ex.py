def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks
left = Sequential()
left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_shape=(99, 13)))
right = Sequential()
right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid', input_shape=(99, 13), go_backwards=True))

model = Sequential()
model.add(Merge([left, right], mode='sum'))

#Add second Bidirectional LSTM layer

left, right = fork(model)

left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid'))

right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', return_sequences=True, activation='tanh',
               inner_activation='sigmoid',  go_backwards=True))

#Rest of the stuff as it is

model = Sequential()
model.add(Merge([left, right], mode='sum'))

model.add(TimeDistributedDense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
print("Train...")
model.fit([X_train, X_train], Y_train, batch_size=1, nb_epoch=nb_epoches, validation_data=([X_test, X_test], Y_test), verbose=1, show_accuracy=True)
