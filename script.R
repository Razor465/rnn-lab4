# підключаємо бібліотеки
library(keras)
library(tensorflow)

num_words <- 10000
max_length <- 1000

data <- dataset_imdb(num_words = num_words) # зчитуємо дані
c(c(x_train, y_train), c(x_test, y_test)) %<-% data

x_train <- pad_sequences(x_train, maxlen = max_length)
x_test <- pad_sequences(x_test, maxlen = max_length)

# embedding

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, output_dim = 16, input_length =  max_length) %>%
  layer_flatten() %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = c('acc'))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)
model %>% evaluate(x_test, y_test)

# rnn

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, output_dim = 8, input_length =  max_length) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = c('acc'))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)
model %>% evaluate(x_test, y_test)

# rnn 2

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, output_dim = 8, input_length =  max_length) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = c('acc'))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)
model %>% evaluate(x_test, y_test)

# rnn mod

model <- keras_model_sequential() %>%
  layer_embedding(input_dim = num_words, output_dim = 8, input_length =  max_length) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_lstm(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = c('acc'))

model %>% fit(x_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2)
model %>% evaluate(x_test, y_test)
