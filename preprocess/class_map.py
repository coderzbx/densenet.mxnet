# -*-coding:utf-8-*_

from collections import namedtuple


Label = namedtuple(
    'Label', ['id', 'categoryId', 'label', 'name'])

arrow_labels = {
    Label(0,    0,      "0",    u"其他"),
    Label(1,    1,      "1",    u"直行标志"),
    Label(2,    2,      "2",    u"前方左转标志"),
    Label(3,    3,      "3",    u"前方右转标志"),
    Label(4,    4,      "4",    u"直行或左转标志"),
    Label(5,    5,      "5",    u"直行或右转标志"),
    Label(6,    6,      "6",    u"前方掉头标志"),
    Label(7,    7,      "7",    u"前方可直行或掉头标志"),
    Label(8,    8,      "8",    u"前方可左转或掉头标志"),
    Label(9,    9,      "9",    u"前方道路仅可左右转弯标志"),
    Label(10,   10,     "10",   u"前方道路有左弯或需向左合流标"),
    Label(11,   11,     "11",   u"前方道路有右弯或需向右合流标"),
    Label(12,   12,     "12",   u"文字"),
    Label(13,   13,     "13",   u"数字"),
    Label(14,   14,     "14",   u"符号"),
}

arrow_labels_v1 = {
    Label(0,    0,      "0",    u"其他"),
    Label(1,    1,      "1",    u"直行标志"),
    Label(2,    1,      "2",    u"前方左转标志"),
    Label(3,    1,      "3",    u"前方右转标志"),
    Label(4,    1,      "4",    u"直行或左转标志"),
    Label(5,    1,      "5",    u"直行或右转标志"),
    Label(6,    1,      "6",    u"前方掉头标志"),
    Label(7,    1,      "7",    u"前方可直行或掉头标志"),
    Label(8,    1,      "8",    u"前方可左转或掉头标志"),
    Label(9,    1,      "9",    u"前方道路仅可左右转弯标志"),
    Label(10,   1,     "10",   u"前方道路有左弯或需向左合流标"),
    Label(11,   1,     "11",   u"前方道路有右弯或需向右合流标"),
    Label(12,   0,     "12",   u"文字"),
    Label(13,   0,     "13",   u"数字"),
    Label(14,   0,     "14",   u"符号"),
}

arrow_labels_v2 = {
    Label(0,    0,      "0",    u"其他"),
    Label(1,    1,      "1",    u"直行标志"),
    Label(2,    2,      "2",    u"前方左转标志"),
    Label(3,    3,      "3",    u"前方右转标志"),
    Label(4,    4,      "4",    u"直行或左转标志"),
    Label(5,    5,      "5",    u"直行或右转标志"),
    Label(6,    6,      "6",    u"前方掉头标志"),
    Label(7,    7,      "7",    u"前方可直行或掉头标志"),
    Label(8,    8,      "8",    u"前方可左转或掉头标志"),
    Label(9,    9,      "9",    u"前方道路仅可左右转弯标志"),
    Label(10,   10,     "10",   u"前方道路有左弯或需向左合流标"),
    Label(11,   11,     "11",   u"前方道路有右弯或需向右合流标"),
    Label(12,   12,     "12",   u"文字"),
    Label(13,   13,     "13",   u"数字"),
    Label(14,   14,     "14",   u"符号"),
}