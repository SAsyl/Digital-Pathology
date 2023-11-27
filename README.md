# Digital-Pathology

Для решения задачи рассматривались класические модель, основанная на свёрточных нейронных сетях. Был применен метод переноса обучения для классификации изображений на 9 классов: 'ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM'.

## Архитектура:
Преобученная модель, которую брали в качетве экстрактора признаков: EfficientNetB0 (https://arxiv.org/abs/1905.11946 — EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks).
Модель, которая на вход принимала призники из EfficientNetB0: batch-нормализация, FC layer с некоторыми видами регуляризации, Dropout слоем и FC слоем с иготовой функцией активации (softmax) для определения вероятности класса.
Метод оптимизации: Adamax.
Функция потерь: categorical_crossentropy.

## Реализованные дополнения:
  * LBL1: Валидация модели на части обучающей выборки.
  * LBL2: Вывод различных показателей в процессе обучения.
  * LBL3: Построение графиков, визуализирующих процесс обучения.
  * LBL4: Использование аугментации и других способов синтетического расширения набора данных.
  * LBL5: Добавлены 2 метода в класс _Dataset_ для детального отображении информации:
   * отображение круговой диаграммы распределения меток,
   * отображение случайного изображения с соответствующей меткой.

## Применение:

Выбор соответствующего датасета:
```python
set_name = 'small'

if set_name == 'tiny':
    d_train = Dataset('train_tiny')
    d_test = Dataset('test_tiny')
elif set_name == 'small':
    d_train = Dataset('train_small')
    d_test = Dataset('test_small')
elif set_name == 'normal':
    d_train = Dataset('train')
    d_test = Dataset('test')

d_train.get_info('Train')
d_test.get_info('Test')


d_train.display_random_image()
```

Инициализация модели, обучение, сохранение и отображение графиков (_Обучать_ if EVALUATE_ONLY == True else _Загрузить модель_):
```python
model = Model()
n_epochs = 10
filename_save = f'{set_name}/{set_name}_{n_epochs}epochs/{set_name}_{n_epochs}epochs'
if EVALUATE_ONLY:
    model.train(d_train, n_epochs=n_epochs, batch_size=8)
    model.plot_hist(f'{filename_save}_AccLoss_plot')
    model.save(filename_save)
else:
    model.load(filename_save)
```

Оценить модель на полной тестовой выборке (TEST_ON_LARGE_DATASET = True): 
```python
if TEST_ON_LARGE_DATASET:
    pred_2 = model.test_on_dataset(d_test)
    Metrics.print_all(d_test.labels, pred_2, 'test')
```

Построение модели и оценивание на загруженном датасете: 
```python
final_model = Model()
final_model.load('best/best')
d_test = Dataset('test')
pred = final_model.test_on_dataset(d_test)
Metrics.print_all(d_test.labels, pred, 'test')
```

Как загрузить лучшую модель и оценить ее на новом датасете ? Модель находится в ./models/best/. Поэтому путь для загрузкиЖ
```python
model_best_weights_filename = 'best/best'
final_model = Model()
final_model.load(model_best_weights_filename)
d_test = Dataset('test')
pred = final_model.test_on_dataset(d_test)
Metrics.print_all(d_test.labels, pred, 'test')
```
