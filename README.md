# Курс глубокого обучения и анализа данных для РТУ МИРЭА

## О курсе

Этот курс предназначен для всех, кто хочет получить базовое представление о машинном обучении и ИИ. Мы начнем с базовых понятий и теорий, чтобы дать вам прочную основу для понимания более сложных тем. Затем мы перейдем к практическим примерам, чтобы помочь вам применить свои знания на практике. В конце курса вы сможете использовать машинное обучение и ИИ для решения реальных задач.

## Преподавание курса

## Содержание курса

### 1 семестр 

№   | Л/П | Название | Ссылка | Комментарии
----|-----|----------|--------|------------
1   |  Л  | Искусственный интеллект и анализ данных | | Определение. Различные направления. История. Этика ИИ.
1.1 |  П  | Обсуждения этики ИИ. | | Тест Тьюринга c GPT. Moral Machine.
1.2 |  П  | Практика работы с ГенИИ | [Notebook](01-Intro/IntroPromptEngineering.ipynb) | Общие понятия о промпт-инжиниринге. Рисуем комикс или пишем рассказ
2   |  Л  | Классическое машинное обучение - 1 | [Regression](02-ML-1/Regression.ipynb), [Classification](02-ML-1/Classification.ipynb)
2.1 |  П  | Пример решения задачи регрессии | [Seminar](02-ML-1/Regression_Seminar.ipynb) 
2.2 |  П  | Пример решения задачи классификации | [Seminar](02-ML-1/Classification_Seminar.ipynb)
3   |  Л  | Классическое машинное обучение - 2 | [Theory](02-ML-2/Clustering.ipynb),
3.1 |  П  | Пример решения задачи кластеризации | [Seminar](02-ML-2/Clustering_Seminar.ipynb), [Lab](02-ML-2/Clustering_Lab.ipynb)
3.2 |  П  | TBD
4   |  Л  | Нейросети. Персептрон Розенблатта. | [Theory](04-NeuralNets/Perceptron.ipynb)
4.1 |  П  | Практика работы с персептроном | [Seminar](04-NeuralNets/Perceptron_Seminar.ipynb)
4.1 |  П  | Строим свой нейросетевой фреймворк | [Seminar](04-NeuralNetworks/Perceptron_Lab.ipynb)
4.2 |  Д  | Экспериментируем с алгоритмами обучения
5   |  Л  | Нейросетевые фреймворки | [Keras](05-NeuralFrameworks/IntroKerasTF.ipynb)
5.1 |  П  | Экспериментируем с Tensorflow/Keras | [Seminar](05-NeuralFrameworks/KerasTF_Seminar.ipynb)
5.2 |  П  | Экспериментируем с PyTorch | [Seminar](05-NeuralFrameworks/PyTorch_Seminar.ipynb)
5.3 |  Д  | Самостоятельное задание | [Задание](05-NeuralFrameworks/Tensorflow_Tasks.ipynb), [Решение](05-NeuralFrameworks/Tensorflow_Tasks_Solved.ipynb)
6   |  Л  | Введение в компьютерное зрение | [Лекция](06-IntroCV/OpenCV.ipynb)
6.1 |  П  | Работа с лицами. Когнитивный портрет. | [Семинар](06-IntroCV/FaceLandmarks.ipynb)
6.2 |  П  | Оптический поток. Обнаружение движения. | [Лекция](06-IntroCV/OpenCV.ipynb)
7   |  Л  | Свёрточные сети | [Лекция](07-ConvNets/ConvolutionNetworks.ipynb)
7.1 |  П  | Обучаем свёрточную сеть "с нуля" | [Задание](07-ConvNets/Faces.ipynb), [Решение](07-ConvNets/Faces_Solution.ipynb)
7.2 |  П  | Визуализируем веса свёрточной сети
8   |  Л  | Transfer Learning
8.1 |  П  | Обучаем сеть на распознавание пород кошек и собак | [Семинар](08-TransferLearning/TransferLearning.ipynb)
8.2 |  П  | Визуализируем активации. GradCam. | [Семинар 1](08-TransferLearning/CNN_Visualzation.ipynb), [Семинар 2](08-TransferLearning/CNN_Visualzation_2.ipynb) 
8.3 |  Д  | Классификация пород кошек и собак | [Задание](08-TransferLearning/Pets.ipynb)

### Второй семестр

№   | Л/П | Название | Ссылка | Комментарии
----|-----|----------|--------|------------
9   |  Л  | Object Detection и Semantic Segmentation
9.1 |  П  | Тренируем сеть Semantic Segmentation | [Семинар](09-DetectionSegmentation/SemanticSegmentation.ipynb)
9.2 |  П  | Тренируем сеть Object Detection | [Семинар](09-DetectionSegmentation/ObjectDetection.ipynb)
10  |  Л  | Генеративные сети и автоэнкодеры | [Лекция](10-AutoEncoders/AutoencodersTF.ipynb)
10.1 |  П  | VAE для понижения размерности | [Семинар](10-AutoEncoders/LongSeminar.ipynb)
10.2 |  П  | GAN на MNIST и CIFAR-10 | [Семинар](10-AutoEncoders/GANTF.ipynb)
11  |  Л  | Основные задачи NLP.
11.1 |  П  | Исследуем возможности NLTK
11.2 |  П  | Анализ статей COVID-19
12  |  Л  | Bag of Words и TF/IDF | [Лекция](12-BoW-TFIDF/TextRepresentation.ipynb)
12.1 |  П  | Анализ тональности текста | [Семинар](12-BoW-TFIDF/TFIDF-Sentiment.ipynb)
12.2 |  П  | Кластеризация новостей на основе TF/IDF
13  |  Л  | Эмбеддинги | [Лекция](13-Embeddings/Embeddings.ipynb)
13.1 |  П  | Тренируем эмбеддинги | [Семинар](13-Embeddings/CBoW.ipynb)
13.2 |  П  | Кластеризация новостей | [Семинар](13-Embeddings/SemanticEmbeddings.ipynb)
14  |  Л  | Рекуррентные нейросети и LSTM | [Лекция](14-RecurrentNets/RNN.ipynb)
14.1 |  П  | Генеративные рекуррентные нейросети | [Семинар](14-RecurrentNets/GenerativeRNN.ipynb)
14.2 |  П  | Решаем задачу NER | [Семинар](14-RecurrentNets/NER.ipynb)
15  |  Л  | Трансформерные архитектуры
15.1 |  П  | Обучаем Transformer с нуля | [Семинар](15-Transformers/Transformers.ipynb)
15.2 |  П  | До-обучаем GPT-2/3 на базе фреймворка HuggingFace transformers | [Семинар](15-Transformers/HuggingFace.ipynb), [Семинар 2](15-Transformers/NER_BERT.ipynb), [Семинар 3](15-Transformers/GPT_Finetune.ipynb)
16  |  Л  | Vision Transformers и мультимодальные сети
16.1 |  П  | Путешествуем по латентному пространству Stable Diffusion | [Семинар](16-MultiModal/StableDiffusionLatentVideo.ipynb)
16.2 |  П  | Обучаем свою сеть Stable Diffusion

## Авторы курса

* [Дмитрий Сошников](http://soshnikov.com), к.ф.-м.н., доцент

В курсе использованы открытые материалы [Microsoft AI for Beginners Curriculum](http://github.com/microsoft/ai-for-beginners) и [Microsoft ML for Beginners Curriculum](http://github.com/microsoft/ML-For-Beginners).
