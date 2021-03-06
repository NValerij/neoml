# Класс CCtcDecodingLayer

<!-- TOC -->

- [Класс CCtcDecodingLayer](#класс-cctcdecodinglayer)
    - [Настройки](#настройки)
        - [Пробел между метками](#пробел-между-метками)
        - [Порог отсечения пробелов](#порог-отсечения-пробелов)
        - [Порог отсечения дуг](#порог-отсечения-дуг)
    - [Обучаемые параметры](#обучаемые-параметры)
    - [Входы](#входы)
    - [Выходы](#выходы)
        - [Наиболее вероятная последовательность](#наиболее-вероятная-последовательность)
        - [Граф линейного деления](#граф-линейного-деления)

<!-- /TOC -->

Класс реализует слой, выполняющий поиск наиболее вероятных последовательностей меток в ответах сети, обученной для задачи [CTC](README.md).

## Настройки

### Пробел между метками

```c++
void SetBlankLabel( int blankLabel );
```

Установка значения метки, которая будет интерпретироваться как пробел между другими метками.

### Порог отсечения пробелов

```c++
void SetBlankProbabilityThreshold( float threshold );
```

Установка порога отсечения пробелов по вероятности при построении графа линейного деления (ГЛД).

### Порог отсечения дуг

```c++
void SetArcProbabilityThreshold( float threshold );
```

Установка порога отсечения дуг по вероятности при построении ГЛД.

## Обучаемые параметры

Слой не имеет обучаемых параметров.

## Входы

Слой имеет один или два входа.

1. На первый вход подаётся блоб с ответами сети размера:
    * `BatchLength` - максимальная длина последовательности ответов
    * `BatchWidth` - количество последовательностей в наборе
    * `ListSize` равен `1`
    * `Height * Width * Depth * Channels` - количество классов
2. *[Опционально]* На второй вход подаётся блоб с данными типа `int`, содержащий длины последовательностей ответов сети. Если этого входа нет, то считается, что длины всех последовательностей ответов сети равны `BatchLength` блоба из первого входа. Блоб должен иметь следующий размер:
    * `BatchWidth` должен быть равен `BatchWidth` блоба из первого входа
    * остальные размерности равны `1`

## Выходы

Слой не имеет выходов.

### Наиболее вероятная последовательность

```c++
void GetBestSequence(int sequenceNumber, CArray<int>& bestLabelSequence) const;
```

Получение наиболее вероятной последовательности для объекта `sequenceNumber` из набора.

### Граф линейного деления

```c++
bool BuildGLD(int sequenceNumber, CLdGraph<CCtcGLDArc>& gld) const;
```

Получение графа линейного деления для объекта `sequenceNumber` из набора.
