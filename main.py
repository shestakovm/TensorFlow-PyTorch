import torch
import torchvision
import torchvision.datasets as dsets
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import numpy as np
from tensorflow.keras import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from google.colab import drive

drive.mount('/content/drive')
df = pd.read_csv('/content/drive/My Drive/neural/pandas/moscow.csv', sep=";")

df = df.iloc[::2, :]
data = df.values


def get_rooms_count(d, max_room_count):
    rooms_count_str = d[0]  # строка с числом комнат

    rooms_count = 0
    try:
        rooms_count = int(rooms_count_str)
        if (rooms_count > max_room_count):
            rooms_сount = max_room_count
    except:
        if (rooms_count_str == rooms_count_str):  # проверка строки на NaN
            if ("Ст" in rooms_count_str):
                rooms_count = max_room_count + 1

    return rooms_count


def get_rooms_count_category(d, max_room_count):
    rooms_count = get_rooms_count(d, max_room_count)
    rooms_count = utils.to_categorical(rooms_count,
                                       max_room_count + 2)  # max_room_count+2 потому что 0 зарезервирован
    # на неопознаное число комнат, а max_room_count+1 на "Студию"
    return rooms_count


def get_metro(d, all_metro_names):
    metro_str = d[1]
    metro = 0

    if (metro_str in all_metro_names):
        metro = all_metro_names.index(metro_str) + 1  # +1 так как 0 зарезервирован на неопознанное метро

    return metro


# Получаем тип метро
# 0 - внутри кольца
# 1 - кольцо
# 2 - 1-3 станции от конца
# 3 - 4-8 станций от кольца
# 4 - больше 8 станций от кольца

def get_metro_type(d):
    metro_type_str = d[1]
    metro_type_classes = 5
    metro_type = metro_type_classes - 1  # изначально считаем последний класс

    # метро внутри кольца
    metro_names_inside_circle = ["Площадь Революции", "Арбатская", "Смоленская", "Красные Ворота", "Чистые пруды",
                                 "Лубянка", "Охотный Ряд", "Библиотека имени Ленина", "Кропоткинская", "Сухаревская",
                                 "Тургеневская", "Китай-город", "Третьяковская", "Трубная", "Сретенский бульвар",
                                 "Цветной бульвар", "Чеховская", "Боровицкая", "Полянка", "Маяковская", "Тверская",
                                 "Театральная", "Новокузнецкая", "Пушкинская", "Кузнецкий Мост", "Китай-город",
                                 "Александровский сад"]

    # метро на кольце
    metro_names_circle = ["Киевская", "Парк Культуры", "Октябрьская", "Добрынинская", "Павелецкая", "Таганская",
                          "Курская", "Комсомольская", "Проспект Мира", "Новослободская", "Белорусская",
                          "Краснопресненская"]

    # метро 1-3 станции от кольца
    metro_names_13_from_circle = ["Бауманская", "Электрозаводская", "Семёновская", "Площадь Ильича", "Авиамоторная",
                                  "Шоссе Энтузиастов", "Римская", "Крестьянская Застава", "Дубровка", "Пролетарская",
                                  "Волгоградский проспект", "Текстильщики", "Автозаводская", "Технопарк", "Коломенская",
                                  "Тульская", "Нагатинская", "Нагорная", "Шаболовская", "Ленинский проспект",
                                  "Академическая", "Фрунзенская", "Спортивная", "Воробьёвы горы", "Студенческая",
                                  "Кутузовская", "Фили", "Парк Победы", "Выставочная", "Международная",
                                  "Улица 1905 года", "Беговая", "Полежаевская", "Динамо", "Аэропорт", "Сокол",
                                  "Деловой центр", "Шелепиха", "Хорошёвская", "ЦСКА", "Петровский парк", "Савёловская",
                                  "Дмитровская", "Тимирязевская", "Достоевская", "Марьина Роща", "Бутырская",
                                  "Фонвизинская", "Рижская", "Алексеевская", "ВДНХ", "Красносельская", "Сокольники",
                                  "Преображенская площадь"]

    # метро 4-8 станций от кольца
    metro_names_48_from_circle = ["Партизанская", "Измайловская", "Первомайская", "Щёлковская", "Новокосино",
                                  "Новогиреево", "Перово", "Кузьминки", "Рязанский проспект", "Выхино",
                                  "Лермонтовский проспект", "Жулебино", "Партизанская", "Измайловская", "Первомайская",
                                  "Щёлковская", "Новокосино", "Новогиреево", "Перово", "Кузьминки",
                                  "Рязанский проспект", "Выхино", "Лермонтовский проспект", "Жулебино",
                                  "Улица Дмитриевского", "Кожуховская", "Печатники", "Волжская", "Люблино",
                                  "Братиславская", "Коломенская", "Каширская", "Кантемировская", "Царицыно", "Орехово",
                                  "Севастопольская", "Чертановская", "Южная", "Пражская", "Варшавская", "Профсоюзная",
                                  "Новые Черёмушки", "Калужская", "Беляево", "Коньково", "Университет",
                                  "Багратионовская", "Филёвский парк", "Пионерская", "Кунцевская", "Молодёжная",
                                  "Октябрьское Поле", "Щукинская", "Спартак", "Тушинская", "Сходненская", "Войковская",
                                  "Водный стадион", "Речной вокзал", "Беломорская", "Ховрино", "Петровско-Разумовская",
                                  "Владыкино", "Отрадное", "Бибирево", "Алтуфьево", "Фонвизинская", "Окружная",
                                  "Верхние Лихоборы", "Селигерская", "ВДНХ", "Ботанический сад", "Свиблово",
                                  "Бабушкинская", "Медведково", "Преображенская площадь", "Черкизовская",
                                  "Бульвар Рокоссовского"]

    if (metro_type_str in metro_names_inside_circle):
        metro_type = 0
    if (metro_type_str in metro_names_circle):
        metro_type = 1
    if (metro_type_str in metro_names_13_from_circle):
        metro_type = 2
    if (metro_type_str in metro_names_48_from_circle):
        metro_type = 3

    metro_type = utils.to_categorical(metro_type, metro_type_classes)
    return metro_type


def get_metro_distance(d):
    metro_distance_str = d[2]
    metro_distance = 0
    metro_distance_type = 0

    if (metro_distance_str == metro_distance_str):
        if (len(metro_distance_str) > 0):

            # определение тип расстояния
            if (metro_distance_str[-1] == "п"):
                metroDistanceType = 1  # пешком
            elif (metro_distance_str[-1] == "т"):
                metroDistanceType = 2  # на транспорте

            # выбрасываем последний символ, чтобы осталось только число
            metro_distance_str = metro_distance_str[:-1]
            try:
                metro_distance = int(metro_distance_str)
                if (metro_distance < 3):
                    metro_distance = 1
                elif (metro_distance < 6):
                    metro_distance = 2
                elif (metro_distance < 10):
                    metro_distance = 3
                elif (metro_distance < 15):
                    metro_distance = 4
                elif (metro_distance < 20):
                    metro_distance = 5
                else:
                    metro_distance = 6
            except:  # если в строке не число, то категория 0
                metro_distance = 0

    metro_distance_classes = 7

    # 7 категорий дистанции по расстоянию
    # 3 типа дистанции - неопознанный, пешком и транспортом
    # создается вектор длины 3*7 = 21
    # преобразую индекс расстояния 0-6 в 0-20
    # Для типа "Пешком" - ничего не меняется
    if (metro_distance_type == 2):
        metro_distance += metro_distance_classes  # Для типа "Транспортом" добавляется 7
    if (metro_distance_type == 0):
        metro_distance += 2 * metro_distance_classes  # Для неопознанного типа добавляется 14

    metro_distance = utils.to_categorical(metro_distance, 3 * metro_distance_classes)
    return metro_distance

    # - этаж квартиры
    # - этажность дома
    # - индикатор, что последний этаж
    # - тип дома


def get_house_type_and_floor(d):
    try:
        house_str = d[3]  # Получаем строку типа дома и этажей
    except:
        house_str = ""

    house_type = 0  # Тип дома
    floor = 0  # Этаж квартиры
    floors = 0  # Этажность дома
    is_last_floor = 0  # Индикатор последнего этажа

    # Проверяем строку на nan
    if (house_str == house_str):
        if (len(house_str) > 1):

            try:
                slash_index = house_str.index("/")  # разделитель /
            except:
                print(house_str)

            try:
                space_index = house_str.index(" ")  # разделитель " "
            except:
                print(house_str)

            floor_str = house_str[:slash_index]
            floors_str = house_str[slash_index + 1:space_index]
            house_type_str = house_str[space_index + 1:]

            try:
                floor = int(floor_str)
                floor_save = floor
                if (floor_save < 5):
                    floor = 2
                if (floor_save < 10):
                    floor = 3
                if (floor_save < 20):
                    floor = 4
                if (floor_save >= 20):
                    floor = 5
                if (floor_save == 1):  # Первый этаж выделяем в отдельную категорию
                    floor = 1

                if (floor == floors):  # Если этаж последний, включаем индикатор последнего этажа
                    is_last_floor = 1
            except:
                floor = 0  # Если строка не парсится в число, то категория этажа = 0 (отдельная)

            # Выбираем категорию этажности дома
            try:
                floors = int(floor_str)
                floors_save = floors
                if (floors_save < 5):
                    floors = 1
                if (floors_save < 10):
                    floors = 2
                if (floors_save < 20):
                    floors = 3
                if (floors_save >= 20):
                    floors = 4
            except:
                floors = 0  # Если строка не парсится в число, то категория этажности = 0 (отдельная)

            # Определяем категорию типа дома
            if (len(house_type_str) > 0):
                if ("М" in house_type_str):
                    house_type = 1
                if ("К" in house_type_str):
                    house_type = 2
                if ("П" in house_type_str):
                    house_type = 3
                if ("Б" in house_type_str):
                    house_type = 4
                if ("?" in house_type_str):
                    house_type = 5
                if ("-" in house_type_str):
                    house_type = 6

        floor = utils.to_categorical(floor, 6)
        floors = utils.to_categorical(floors, 5)
        house_type = utils.to_categorical(house_type, 7)

    return floor, floors, is_last_floor, house_type


def get_balcony(d):
    balcony_str = d[4]

    balcony_variants = ['Л', 'Б', '2Б', '-', '2Б2Л', 'БЛ', '3Б', '2Л', 'Эрк', 'Б2Л', 'ЭркЛ', '3Л', '4Л', '*Л', '*Б']

    if (balcony_str == balcony_str):
        balcony = balcony_variants.index(balcony_str) + 1
    else:
        balcony = 0

    balcony = utils.to_categorical(balcony, 16)

    return balcony


def get_wc(d):
    wc_str = d[5]

    wc_variants = ['2', 'Р', 'С', '-', '2С', '+', '4Р', '2Р', '3С', '4С', '4', '3', '3Р']

    if (wc_str == wc_str):
        wc = wc_variants.index(wc_str) + 1
    else:
        wc = 0

    wc = utils.to_categorical(wc, 14)

    return wc


def get_area(d):
    area_str = d[6]

    if ("/" in area_str):
        slash_index = area_str.index("/")
        try:
            area = float(area_str[:slash_index])
        except:
            area = 0
    else:
        area = 0

    return area


def get_cost(d):
    cost_str = d[7]

    try:
        cost = float(cost_str)
    except:
        cost = 0

    return cost


def get_comment(d):
    comment_str = d[-1]

    return comment_str


def get_all_parameters(d, all_metro_names):
    rooms_count_type = get_rooms_count_category(d, 30)
    metro = get_metro(d, all_metro_names)
    metro_type = get_metro_type(d)
    metro_distance = get_metro_distance(d)
    floor, floors, is_last_floor, house_type = get_house_type_and_floor(d)
    balcony = get_balcony(d)
    wc = get_wc(d)
    area = get_area(d)

    # Объединяем в один лист
    out = list(rooms_count_type)
    out.append(metro)
    out.extend(metro_type)
    out.extend(metro_distance)
    out.extend(floor)
    out.extend(floors)
    out.append(is_last_floor)
    out.extend(house_type)
    out.extend(balcony)
    out.extend(wc)
    out.append(area)

    return out


def get_x_train(data):
    all_metro_names = list(df["Метро / ЖД станции"].unique())

    # все строки в data1 в векторы параметров и записано в xTrain
    x_Train = [get_all_parameters(d, all_metro_names) for d in data]
    x_Train = np.array(x_Train)

    return x_Train


def get_y_train(data):
    # Загружаем лист всех цен квартир по всем строкам data1
    cost_list = [get_cost(d) for d in data]
    y_Train = np.array(cost_list)

    return y_Train


one_room_mask = [get_rooms_count(d, 30) == 1 for d in data]  # Делаем маску однокомнатных квартир, принцип (get_rooms_count(d, 30) == 1)
data1 = data[one_room_mask]  # В data1 оставляем только однокомнатные квартиры

x_train = get_x_train(data1)
y_train = get_y_train(data1)


def text_to_words(text):
    text = text.replace(".", "")
    text = text.replace("—", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.replace("?", "")
    text = text.replace("…", "")
    text = text.lower()

    words = []
    curr_word = ""

    for symbol in text:

        if (symbol != "\ufeff"):
            if (symbol != " "):
                curr_word += symbol
            else:
                if (curr_word != ""):
                    words.append(curr_word)
                    curr_word = ""

                    # добавление финального слова, если оно не пустое
    # текст чаще всего заканчивается на не пробел то теряется финальное слово
    if (curr_word != ""):
        words.append(curr_word)

    return words


def create_vocabulary(all_words):
    w_count = dict.fromkeys(all_words, 0)

    for word in all_words:
        w_count[word] += 1

    words_list = list(w_count.items())

    words_list.sort(key=lambda i: i[1], reverse=1)
    # key = lambda i:i[1] - говорит, что сортировать надо по частоте появления
    # в i[0] слово, в i[1] - частота появления
    # reverse=1 говорит сортироваться по убыванию

    sorted_words = []

    for word in words_list:
        sorted_words.append(word[0])

    word_indexes = dict.fromkeys(all_words, 0)

    for word in word_indexes.keys():
        word_indexes[word] = sorted_words.index(word) + 1  # индекс 0 резервируем под неопознанные слова

    return word_indexes


def words_to_indexes(words, vocabulary, max_words_count):
    words_indexes = []

    for word in words:

        word_index = 0
        word_in_vocabulary = word in vocabulary

        if (word_in_vocabulary):
            index = vocabulary[word]  # индекс = индексу слова в словаре
            if (index < max_words_count):  # Если индекс ниже maxWordsCount - черты отсечения слов
                word_index = index  # То записываем индекс
                # Иначе останется значение 0

        words_indexes.append(word_index)

    return words_indexes


def change_x_to01(train_vector, words_count):
    out = np.zeros(words_count)

    for x in train_vector:
        out[x] = 1

    return out


def change_set_to01(train_set, words_count):
    out = []

    for x in train_set:
        out.append(change_x_to01(x, words_count))

    return np.array(out)


def get_x_train_comments(data):
    x_train_comments = []  # Тут будет обучающся выборка
    all_text_comments = ""  # Тут будуте все тексты вместе для словаря

    # Идём по всем строкам квартир в базе
    for d in data:
        curr_text = get_comment(d)  # Вытаскиваем примечание к квартире
        try:
            if (curr_text == curr_text):  # Проверяем на nan
                all_text_comments += curr_text + " "  # Добавляем текст в общий текст для словаря
        except:
            curr_text = "Нет комментария"  # Если не получается, то делаем стандартный текст "Нет комментария"
        x_train_comments.append(curr_text)  # Добавляем примечание новой строкой в обучающую выборку

    x_train_comments = np.array(x_train_comments)

    return (x_train_comments, all_text_comments)


###########################
# Формируем обучающую выборку из примечаний к квартирам
# Теперь в виде индексов
##########################
def change_set_to_indexes(x_train_comments, vocabulary, max_words_count):
    x_train_comments_indexes = []  # Тут будет итоговый x_train примечаний в виде индексов

    # Идём по всем текстам
    for text in x_train_comments:
        curr_words = text_to_words(text)  # Разбиваем текст на слова
        curr_indexes = words_to_indexes(curr_words, vocabulary, max_words_count)  # Превращаем в лист индексов
        curr_indexes = np.array(curr_indexes)
        x_train_comments_indexes.append(curr_indexes)  # Добавляем в x_train

    x_train_comments_indexes = np.array(x_train_comments_indexes)
    x_train_comments_indexes = change_set_to01(x_train_comments_indexes,
                                               max_words_count)  # Превращаем в формат bag of words
    return x_train_comments_indexes


###########################
# Формируем обучающую выборку из примечаний к квартирам
# Теперь в виде индексов
# И с приведением к стандартной длине всех векторов - cropLen
##########################
def change_set_to_indexes_crop(x_train_comments, vocabulary, max_words_count, crop_len):
    x_train_comments_indexes = []  # Тут будет итоговый x_train примечаний в виде индексов

    # Идём по всем текстам
    for text in x_train_comments:
        curr_words = text_to_words(text)  # Разбиваем текст на слова
        curr_indexes = words_to_indexes(curr_words, vocabulary, max_words_count)  # Превращаем в лист индексов
        curr_indexes = np.array(curr_indexes)
        x_train_comments_indexes.append(curr_indexes)  # Добавляем в x_train

    x_train_comments_indexes = np.array(x_train_comments_indexes)
    x_train_comments_indexes = pad_sequences(x_train_comments_indexes,
                                             maxlen=crop_len)  # Приводим все вектора к стандартной длине
    return x_train_comments_indexes


x_trainC, all_text_comments = get_x_train_comments(
    data1)  # создаётся обучающая выборка по текстам и большой текст для словаря
all_words = text_to_words(all_text_comments)  # собирается полный текст в слова
all_words = all_words[::10]  # берется 10% слов (иначе словарь слишком долго формируется)
vocabulary = create_vocabulary(all_words)  # создаём словарь
x_trainC01 = change_set_to_indexes(x_trainC, vocabulary, 2000)  # Преобразеум x_train в bag of words

x_scaler = StandardScaler()  # нормальное распределение
x_scaler.fit(x_train[:, -1].reshape(-1, 1))  # обучается на площадях квартир (последня колонка в x_train)
x_train_scaled = x_train.copy()
x_train_scaled[:, -1] = x_scaler.transform(x_train[:, -1].reshape(-1, 1)).flatten()  # Нормируем данные нормировщиком

y_scaler = StandardScaler()  # Делаемнормальный нормировщик
y_scaler.fit(y_train.reshape(-1, 1))  # Обучаем на ценах квартир
y_train_scaled = y_scaler.transform(y_train.reshape(-1, 1))  # Нормируем цены квартир

x_train, x_test, y_train, y_test = train_test_split(x_train_scaled, y_train_scaled, test_size=0.2, random_state=42)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

input_size = 109
hidden = 200
batch_size = 100
lr = 0.0001
epochs = 100

def create_nn(batch_size=batch_size, lr=lr, epochs=epochs):
    train_data = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, num_workers=4)


    class Regression(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.3)
            self.layer2 = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out = self.layer1(x)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.layer2(out)
            return out


    model = Regression(input_size, hidden)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    losses = []
    model.train()

    for epoch in range(epochs):
     loss_total = 0
     batch = 1
     for (data, labels) in train_loader:
       label = labels.type(torch.FloatTensor)
       optimizer.zero_grad()
       outputs = model(data)
       output = outputs.type(torch.FloatTensor)
       loss = criterion(output, label)
       loss.backward()

       loss_total += loss.detach().data

       pred_unscaled = y_scaler.inverse_transform(model(x_test[0:100]).detach()).flatten()
       y_train_unscaled = y_scaler.inverse_transform(y_test[0:100]).flatten()
       delta = pred_unscaled - y_train_unscaled
       abs_delta = abs(delta)
       price_loss = round(sum(abs_delta) / (1e+6 * len(abs_delta)), 3)

       optimizer.step()
       batch +=1

     losses.append(loss_total/len(train_data))
     print('Эпоха: [%d/%d], Ошибка: %.4f. Ошибка в млн рублей: %.4f' % (epoch+1, epochs, loss, price_loss))

  if __name__ == "__main__":
    create_nn()