import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import geopandas as gpd
import odc.stac
import pandas as pd
import planetary_computer
import rioxarray
import xarray as xr
from pystac.item import Item
from tqdm import tqdm

from environmental_risk_metrics.base import BaseEnvironmentalMetric
from environmental_risk_metrics.utils.planetary_computer import (
    get_planetary_computer_items,
)

logger = logging.getLogger(name=__name__)


OPENLANDMAP_LC = {
    "2000": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20000101_20001231_go_epsg.4326_v20230901.tif",
    "2005": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20050101_20051231_go_epsg.4326_v20230901.tif",
    "2010": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20100101_20101231_go_epsg.4326_v20230901.tif",
    "2015": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20150101_20151231_go_epsg.4326_v20230901.tif",
    "2020": "https://s3.openlandmap.org/arco/lc_glad.glcluc_c_30m_s_20200101_20201231_go_epsg.4326_v20230901.tif",
}

ESA_LAND_COVERclass_conversion_dict = {
    0: "No data",
    10: "Cropland, rainfed",
    11: "Cropland, rainfed, herbaceous cover",
    12: "Cropland, rainfed, tree, or shrub cover",
    20: "Cropland, irrigated or post-flooding",
    30: "Mosaic cropland (>50%) / natural vegetation (tree, shrub, herbaceous cover) (<50%)",
    40: "Mosaic natural vegetation (tree, shrub, herbaceous cover) (>50%) / cropland (<50%)",
    50: "Tree cover, broadleaved, evergreen, closed to open (>15%)",
    60: "Tree cover, broadleaved, deciduous, closed to open (>15%)",
    61: "Tree cover, broadleaved, deciduous, closed (>40%)",
    62: "Tree cover, broadleaved, deciduous, open (15-40%)",
    70: "Tree cover, needleleaved, evergreen, closed to open (>15%)",
    71: "Tree cover, needleleaved, evergreen, closed (>40%)",
    72: "Tree cover, needleleaved, evergreen, open (15-40%)",
    80: "Tree cover, needleleaved, deciduous, closed to open (>15%)",
    81: "Tree cover, needleleaved, deciduous, closed (>40%)",
    82: "Tree cover, needleleaved, deciduous, open (15-40%)",
    90: "Tree cover, mixed leaf type (broadleaved and needleleaved)",
    100: "Mosaic tree and shrub (>50%) / herbaceous cover (<50%)",
    110: "Mosaic herbaceous cover (>50%) / tree and shrub (<50%)",
    120: "Shrubland",
    121: "Evergreen shrubland",
    122: "Deciduous shrubland",
    130: "Grassland",
    140: "Lichens and mosses",
    150: "Sparse vegetation (tree, shrub, herbaceous cover) (<15%)",
    151: "Sparse tree (<15%)",
    152: "Sparse shrub (<15%)",
    153: "Sparse herbaceous cover (<15%)",
    160: "Tree cover, flooded, fresh or brackish water",
    170: "Tree cover, flooded, saline water",
    180: "Shrub or herbaceous cover, flooded, fresh/saline/brackish water",
    190: "Urban areas",
    200: "Bare areas",
    201: "Consolidated bare areas",
    202: "Unconsolidated bare areas",
    210: "Water bodies",
    220: "Permanent snow and ice",
}

ESRI_LAND_COVERclass_conversion_dict = {
    0: "No Data",
    1: "Water",
    2: "Trees",
    4: "Flooded vegetation",
    5: "Crops",
    7: "Built area",
    8: "Bare ground",
    9: "Snow/ice",
    10: "Clouds",
    11: "Rangeland",
}


GLAD_LAND_COVERclass_conversion_dict = {
    1: "Terra Firma, true desert - 7% short veg. cover (1)",
    2: "Terra Firma, semi-arid - 11% short veg. cover (2)",
    3: "Terra Firma, semi-arid - 15% short veg. cover (3)",
    4: "Terra Firma, semi-arid - 19% short veg. cover (4)",
    5: "Terra Firma, semi-arid - 23% short veg. cover (5)",
    6: "Terra Firma, semi-arid - 27% short veg. cover (6)",
    7: "Terra Firma, semi-arid - 31% short veg. cover (7)",
    8: "Terra Firma, semi-arid - 35% short veg. cover (8)",
    9: "Terra Firma, semi-arid - 39% short veg. cover (9)",
    10: "Terra Firma, semi-arid - 43% short veg. cover (10)",
    11: "Terra Firma, semi-arid - 47% short veg. cover (11)",
    12: "Terra Firma, semi-arid - 51% short veg. cover (12)",
    13: "Terra Firma, semi-arid - 55% short veg. cover (13)",
    14: "Terra Firma, semi-arid - 59% short veg. cover (14)",
    15: "Terra Firma, semi-arid - 63% short veg. cover (15)",
    16: "Terra Firma, semi-arid - 67% short veg. cover (16)",
    17: "Terra Firma, semi-arid - 71% short veg. cover (17)",
    18: "Terra Firma, semi-arid - 75% short veg. cover (18)",
    19: "Terra Firma, dense short vegetation - 79% short veg. cover (19)",
    20: "Terra Firma, dense short vegetation - 83% short veg. cover (20)",
    21: "Terra Firma, dense short vegetation - 87% short veg. cover (21)",
    22: "Terra Firma, dense short vegetation - 91% short veg. cover (22)",
    23: "Terra Firma, dense short vegetation - 95% short veg. cover (23)",
    24: "Terra Firma, dense short vegetation - 100% short veg. cover (24)",
    25: "Terra Firma, stable tree cover - 3m trees (25)",
    26: "Terra Firma, stable tree cover - 4m trees (26)",
    27: "Terra Firma, stable tree cover - 5m trees (27)",
    28: "Terra Firma, stable tree cover - 6m trees (28)",
    29: "Terra Firma, stable tree cover - 7m trees (29)",
    30: "Terra Firma, stable tree cover - 8m trees (30)",
    31: "Terra Firma, stable tree cover - 9m trees (31)",
    32: "Terra Firma, stable tree cover - 10m trees (32)",
    33: "Terra Firma, stable tree cover - 11m trees (33)",
    34: "Terra Firma, stable tree cover - 12m trees (34)",
    35: "Terra Firma, stable tree cover - 13m trees (35)",
    36: "Terra Firma, stable tree cover - 14m trees (36)",
    37: "Terra Firma, stable tree cover - 15m trees (37)",
    38: "Terra Firma, stable tree cover - 16m trees (38)",
    39: "Terra Firma, stable tree cover - 17m trees (39)",
    40: "Terra Firma, stable tree cover - 18m trees (40)",
    41: "Terra Firma, stable tree cover - 19m trees (41)",
    42: "Terra Firma, stable tree cover - 20m trees (42)",
    43: "Terra Firma, stable tree cover - 21m trees (43)",
    44: "Terra Firma, stable tree cover - 22m trees (44)",
    45: "Terra Firma, stable tree cover - 23m trees (45)",
    46: "Terra Firma, stable tree cover - 24m trees (46)",
    47: "Terra Firma, stable tree cover - 25m trees (47)",
    48: "Terra Firma, stable tree cover - >25m trees (48)",
    49: "Terra Firma, tree cover with prev. disturb. (2020 height) - 3m trees (49)",
    50: "Terra Firma, tree cover with prev. disturb. (2020 height) - 4m trees (50)",
    51: "Terra Firma, tree cover with prev. disturb. (2020 height) - 5m trees (51)",
    52: "Terra Firma, tree cover with prev. disturb. (2020 height) - 6m trees (52)",
    53: "Terra Firma, tree cover with prev. disturb. (2020 height) - 7m trees (53)",
    54: "Terra Firma, tree cover with prev. disturb. (2020 height) - 8m trees (54)",
    55: "Terra Firma, tree cover with prev. disturb. (2020 height) - 9m trees (55)",
    56: "Terra Firma, tree cover with prev. disturb. (2020 height) - 10m trees (56)",
    57: "Terra Firma, tree cover with prev. disturb. (2020 height) - 11m trees (57)",
    58: "Terra Firma, tree cover with prev. disturb. (2020 height) - 12m trees (58)",
    59: "Terra Firma, tree cover with prev. disturb. (2020 height) - 13m trees (59)",
    60: "Terra Firma, tree cover with prev. disturb. (2020 height) - 14m trees (60)",
    61: "Terra Firma, tree cover with prev. disturb. (2020 height) - 15m trees (61)",
    62: "Terra Firma, tree cover with prev. disturb. (2020 height) - 16m trees (62)",
    63: "Terra Firma, tree cover with prev. disturb. (2020 height) - 17m trees (63)",
    64: "Terra Firma, tree cover with prev. disturb. (2020 height) - 18m trees (64)",
    65: "Terra Firma, tree cover with prev. disturb. (2020 height) - 19m trees (65)",
    66: "Terra Firma, tree cover with prev. disturb. (2020 height) - 20m trees (66)",
    67: "Terra Firma, tree cover with prev. disturb. (2020 height) - 21m trees (67)",
    68: "Terra Firma, tree cover with prev. disturb. (2020 height) - 22m trees (68)",
    69: "Terra Firma, tree cover with prev. disturb. (2020 height) - 23m trees (69)",
    70: "Terra Firma, tree cover with prev. disturb. (2020 height) - 24m trees (70)",
    71: "Terra Firma, tree cover with prev. disturb. (2020 height) - 25m trees (71)",
    72: "Terra Firma, tree cover with prev. disturb. (2020 height) - >25m trees (72)",
    73: "Terra Firma, tree height gain (2020 height) - 3m trees (73)",
    74: "Terra Firma, tree height gain (2020 height) - 4m trees (74)",
    75: "Terra Firma, tree height gain (2020 height) - 5m trees (75)",
    76: "Terra Firma, tree height gain (2020 height) - 6m trees (76)",
    77: "Terra Firma, tree height gain (2020 height) - 7m trees (77)",
    78: "Terra Firma, tree height gain (2020 height) - 8m trees (78)",
    79: "Terra Firma, tree height gain (2020 height) - 9m trees (79)",
    80: "Terra Firma, tree height gain (2020 height) - 10m trees (80)",
    81: "Terra Firma, tree height gain (2020 height) - 11m trees (81)",
    82: "Terra Firma, tree height gain (2020 height) - 12m trees (82)",
    83: "Terra Firma, tree height gain (2020 height) - 13m trees (83)",
    84: "Terra Firma, tree height gain (2020 height) - 14m trees (84)",
    85: "Terra Firma, tree height gain (2020 height) - 15m trees (85)",
    86: "Terra Firma, tree height gain (2020 height) - 16m trees (86)",
    87: "Terra Firma, tree height gain (2020 height) - 17m trees (87)",
    88: "Terra Firma, tree height gain (2020 height) - 18m trees (88)",
    89: "Terra Firma, tree height gain (2020 height) - 19m trees (89)",
    90: "Terra Firma, tree height gain (2020 height) - 20m trees (90)",
    91: "Terra Firma, tree height gain (2020 height) - 21m trees (91)",
    92: "Terra Firma, tree height gain (2020 height) - 22m trees (92)",
    93: "Terra Firma, tree height gain (2020 height) - 23m trees (93)",
    94: "Terra Firma, tree height gain (2020 height) - 24m trees (94)",
    95: "Terra Firma, tree height gain (2020 height) - 25m trees (95)",
    96: "Terra Firma, tree height gain (2020 height) - >25m trees (96)",
    100: "Wetland, salt pan - 3% short veg. cover (100)",
    101: "Wetland, salt pan - 7% short veg. cover (101)",
    102: "Wetland, sparse vegetation - 11% short veg. cover (102)",
    103: "Wetland, sparse vegetation - 15% short veg. cover (103)",
    104: "Wetland, sparse vegetation - 19% short veg. cover (104)",
    105: "Wetland, sparse vegetation - 23% short veg. cover (105)",
    106: "Wetland, sparse vegetation - 27% short veg. cover (106)",
    107: "Wetland, sparse vegetation - 31% short veg. cover (107)",
    108: "Wetland, sparse vegetation - 35% short veg. cover (108)",
    109: "Wetland, sparse vegetation - 39% short veg. cover (109)",
    110: "Wetland, sparse vegetation - 43% short veg. cover (110)",
    111: "Wetland, sparse vegetation - 47% short veg. cover (111)",
    112: "Wetland, sparse vegetation - 51% short veg. cover (112)",
    113: "Wetland, sparse vegetation - 55% short veg. cover (113)",
    114: "Wetland, sparse vegetation - 59% short veg. cover (114)",
    115: "Wetland, sparse vegetation - 63% short veg. cover (115)",
    116: "Wetland, sparse vegetation - 67% short veg. cover (116)",
    117: "Wetland, sparse vegetation - 71% short veg. cover (117)",
    118: "Wetland, sparse vegetation - 75% short veg. cover (118)",
    119: "Wetland, dense short vegetation - 79% short veg. cover (119)",
    120: "Wetland, dense short vegetation - 83% short veg. cover (120)",
    121: "Wetland, dense short vegetation - 87% short veg. cover (121)",
    122: "Wetland, dense short vegetation - 91% short veg. cover (122)",
    123: "Wetland, dense short vegetation - 95% short veg. cover (123)",
    124: "Wetland, dense short vegetation - 100% short veg. cover (124)",
    125: "Wetland, stable tree cover - 3m trees (125)",
    126: "Wetland, stable tree cover - 4m trees (126)",
    127: "Wetland, stable tree cover - 5m trees (127)",
    128: "Wetland, stable tree cover - 6m trees (128)",
    129: "Wetland, stable tree cover - 7m trees (129)",
    130: "Wetland, stable tree cover - 8m trees (130)",
    131: "Wetland, stable tree cover - 9m trees (131)",
    132: "Wetland, stable tree cover - 10m trees (132)",
    133: "Wetland, stable tree cover - 11m trees (133)",
    134: "Wetland, stable tree cover - 12m trees (134)",
    135: "Wetland, stable tree cover - 13m trees (135)",
    136: "Wetland, stable tree cover - 14m trees (136)",
    137: "Wetland, stable tree cover - 15m trees (137)",
    138: "Wetland, stable tree cover - 16m trees (138)",
    139: "Wetland, stable tree cover - 17m trees (139)",
    140: "Wetland, stable tree cover - 18m trees (140)",
    141: "Wetland, stable tree cover - 19m trees (141)",
    142: "Wetland, stable tree cover - 20m trees (142)",
    143: "Wetland, stable tree cover - 21m trees (143)",
    144: "Wetland, stable tree cover - 22m trees (144)",
    145: "Wetland, stable tree cover - 23m trees (145)",
    146: "Wetland, stable tree cover - 24m trees (146)",
    147: "Wetland, stable tree cover - 25m trees (147)",
    148: "Wetland, stable tree cover - >25m trees (148)",
    149: "Wetland, tree cover with prev. disturb. (2020 height) - 3m trees (149)",
    150: "Wetland, tree cover with prev. disturb. (2020 height) - 4m trees (150)",
    151: "Wetland, tree cover with prev. disturb. (2020 height) - 5m trees (151)",
    152: "Wetland, tree cover with prev. disturb. (2020 height) - 6m trees (152)",
    153: "Wetland, tree cover with prev. disturb. (2020 height) - 7m trees (153)",
    154: "Wetland, tree cover with prev. disturb. (2020 height) - 8m trees (154)",
    155: "Wetland, tree cover with prev. disturb. (2020 height) - 9m trees (155)",
    156: "Wetland, tree cover with prev. disturb. (2020 height) - 10m trees (156)",
    157: "Wetland, tree cover with prev. disturb. (2020 height) - 11m trees (157)",
    158: "Wetland, tree cover with prev. disturb. (2020 height) - 12m trees (158)",
    159: "Wetland, tree cover with prev. disturb. (2020 height) - 13m trees (159)",
    160: "Wetland, tree cover with prev. disturb. (2020 height) - 14m trees (160)",
    161: "Wetland, tree cover with prev. disturb. (2020 height) - 15m trees (161)",
    162: "Wetland, tree cover with prev. disturb. (2020 height) - 16m trees (162)",
    163: "Wetland, tree cover with prev. disturb. (2020 height) - 17m trees (163)",
    164: "Wetland, tree cover with prev. disturb. (2020 height) - 18m trees (164)",
    165: "Wetland, tree cover with prev. disturb. (2020 height) - 19m trees (165)",
    166: "Wetland, tree cover with prev. disturb. (2020 height) - 20m trees (166)",
    167: "Wetland, tree cover with prev. disturb. (2020 height) - 21m trees (167)",
    168: "Wetland, tree cover with prev. disturb. (2020 height) - 22m trees (168)",
    169: "Wetland, tree cover with prev. disturb. (2020 height) - 23m trees (169)",
    170: "Wetland, tree cover with prev. disturb. (2020 height) - 24m trees (170)",
    171: "Wetland, tree cover with prev. disturb. (2020 height) - 25m trees (171)",
    172: "Wetland, tree cover with prev. disturb. (2020 height) - >25m trees (172)",
    173: "Wetland, tree height gain (2020 height) - 3m trees (173)",
    174: "Wetland, tree height gain (2020 height) - 4m trees (174)",
    175: "Wetland, tree height gain (2020 height) - 5m trees (175)",
    176: "Wetland, tree height gain (2020 height) - 6m trees (176)",
    177: "Wetland, tree height gain (2020 height) - 7m trees (177)",
    178: "Wetland, tree height gain (2020 height) - 8m trees (178)",
    179: "Wetland, tree height gain (2020 height) - 9m trees (179)",
    180: "Wetland, tree height gain (2020 height) - 10m trees (180)",
    181: "Wetland, tree height gain (2020 height) - 11m trees (181)",
    182: "Wetland, tree height gain (2020 height) - 12m trees (182)",
    183: "Wetland, tree height gain (2020 height) - 13m trees (183)",
    184: "Wetland, tree height gain (2020 height) - 14m trees (184)",
    185: "Wetland, tree height gain (2020 height) - 15m trees (185)",
    186: "Wetland, tree height gain (2020 height) - 16m trees (186)",
    187: "Wetland, tree height gain (2020 height) - 17m trees (187)",
    188: "Wetland, tree height gain (2020 height) - 18m trees (188)",
    189: "Wetland, tree height gain (2020 height) - 19m trees (189)",
    190: "Wetland, tree height gain (2020 height) - 20m trees (190)",
    191: "Wetland, tree height gain (2020 height) - 21m trees (191)",
    192: "Wetland, tree height gain (2020 height) - 22m trees (192)",
    193: "Wetland, tree height gain (2020 height) - 23m trees (193)",
    194: "Wetland, tree height gain (2020 height) - 24m trees (194)",
    195: "Wetland, tree height gain (2020 height) - 25m trees (195)",
    196: "Wetland, tree height gain (2020 height) - >25m trees (196)",
    208: "Open surface water, permanent water (208)",
    209: "Open surface water, persistent water loss (209)",
    210: "Open surface water, persistent water gain (210)",
    211: "Open surface water, variable water (211)",
    240: "Short veg. after tree loss (240)",
    241: "Snow/ice, stable (241)",
    242: "Snow/ice, gain (242)",
    243: "Snow/ice, loss (243)",
    244: "Cropland, Stable (244)",
    245: "Cropland, gain from trees (245)",
    246: "Cropland, gain from wetland veg (246)",
    247: "Cropland, gain from other (247)",
    248: "Cropland, loss to tree (248)",
    249: "Cropland, loss to short veg/other (249)",
    250: "Built-up, stable built-up (250)",
    251: "Built-up, gain from trees (251)",
    252: "Built-up, gain from crop (252)",
    253: "Built-up, gain from other (253)",
    254: "Ocean (254)",
    255: "No data (255)",
}


def map_esa_to_esri_classes() -> Optional[int]:
    """Maps ESA land cover classes to ESRI land cover classes"""
    mapping = {
        # ESA 'No data' -> ESRI 'No Data'
        0: 0,
        # ESA 'Cropland, rainfed' -> ESRI 'Crops'
        10: 5,
        # ESA 'Cropland, rainfed, herbaceous cover' -> ESRI 'Crops'
        11: 5,
        # ESA 'Cropland, rainfed, tree, or shrub cover' -> ESRI 'Crops'
        12: 5,
        # ESA 'Cropland, irrigated or post-flooding' -> ESRI 'Crops'
        20: 5,
        # ESA 'Mosaic cropland/natural vegetation' -> ESRI 'Crops'
        30: 5,
        # ESA 'Mosaic natural vegetation/cropland' -> ESRI 'Rangeland'
        40: 11,
        # ESA 'Tree cover, broadleaved, evergreen' -> ESRI 'Trees'
        50: 2,
        # ESA 'Tree cover, broadleaved, deciduous' -> ESRI 'Trees'
        60: 2,
        # ESA 'Tree cover, broadleaved, deciduous, closed' -> ESRI 'Trees'
        61: 2,
        # ESA 'Tree cover, broadleaved, deciduous, open' -> ESRI 'Trees'
        62: 2,
        # ESA 'Tree cover, needleleaved, evergreen' -> ESRI 'Trees'
        70: 2,
        # ESA 'Tree cover, needleleaved, evergreen, closed' -> ESRI 'Trees'
        71: 2,
        # ESA 'Tree cover, needleleaved, evergreen, open' -> ESRI 'Trees'
        72: 2,
        # ESA 'Tree cover, needleleaved, deciduous' -> ESRI 'Trees'
        80: 2,
        # ESA 'Tree cover, needleleaved, deciduous, closed' -> ESRI 'Trees'
        81: 2,
        # ESA 'Tree cover, needleleaved, deciduous, open' -> ESRI 'Trees'
        82: 2,
        # ESA 'Tree cover, mixed leaf type' -> ESRI 'Trees'
        90: 2,
        # ESA 'Mosaic tree and shrub/herbaceous cover' -> ESRI 'Rangeland'
        100: 11,
        # ESA 'Mosaic herbaceous cover/tree and shrub' -> ESRI 'Rangeland'
        110: 11,
        # ESA 'Shrubland' -> ESRI 'Rangeland'
        120: 11,
        # ESA 'Evergreen shrubland' -> ESRI 'Rangeland'
        121: 11,
        # ESA 'Deciduous shrubland' -> ESRI 'Rangeland'
        122: 11,
        # ESA 'Grassland' -> ESRI 'Rangeland'
        130: 11,
        # ESA 'Lichens and mosses' -> ESRI 'Rangeland'
        140: 11,
        # ESA 'Sparse vegetation' -> ESRI 'Rangeland'
        150: 11,
        # ESA 'Sparse tree' -> ESRI 'Rangeland'
        151: 11,
        # ESA 'Sparse shrub' -> ESRI 'Rangeland'
        152: 11,
        # ESA 'Sparse herbaceous cover' -> ESRI 'Rangeland'
        153: 11,
        # ESA 'Tree cover, flooded, fresh/brackish' -> ESRI 'Flooded vegetation'
        160: 4,
        # ESA 'Tree cover, flooded, saline water' -> ESRI 'Flooded vegetation'
        170: 4,
        # ESA 'Shrub or herbaceous cover, flooded' -> ESRI 'Flooded vegetation'
        180: 4,
        # ESA 'Urban areas' -> ESRI 'Built area'
        190: 7,
        # ESA 'Bare areas' -> ESRI 'Bare ground'
        200: 8,
        # ESA 'Consolidated bare areas' -> ESRI 'Bare ground'
        201: 8,
        # ESA 'Unconsolidated bare areas' -> ESRI 'Bare ground'
        202: 8,
        # ESA 'Water bodies' -> ESRI 'Water'
        210: 1,
        # ESA 'Permanent snow and ice' -> ESRI 'Snow/ice'
        220: 9,
    }
    NEW_ESA_CLASS_MAPPING = {}
    for key, value in mapping.items():
        if value is not None:
            NEW_ESA_CLASS_MAPPING[key] = ESRI_LAND_COVERclass_conversion_dict[value]
    return NEW_ESA_CLASS_MAPPING


def map_glad_to_esri_classes() -> Optional[int]:
    """Maps GLAD land cover classes to ESRI land cover classes"""
    GLAD_TO_CLASSES = {
        # Terra Firma short vegetation (1-24)
        **{i: 11 for i in range(1, 25)},
        # Terra Firma stable tree cover (25-48)
        **{i: 2 for i in range(25, 49)},
        # Terra Firma tree cover with prev. disturb. (49-72)
        **{i: 2 for i in range(49, 73)},
        # Terra Firma tree height gain (73-96)
        **{i: 2 for i in range(73, 97)},
        # Wetland short vegetation (100-124)
        **{i: 4 for i in range(100, 125)},
        # Wetland stable tree cover (125-148)
        **{i: 4 for i in range(125, 149)},
        # Wetland tree cover with prev. disturb. (149-172)
        **{i: 4 for i in range(149, 173)},
        # Wetland tree height gain (173-196)
        **{i: 4 for i in range(173, 197)},
        # Open surface water (208-211)
        **{i: 1 for i in range(208, 212)},
        # Short veg. after tree loss (240)
        240: 11,
        # Snow/ice stable/gain/loss (241-243)
        **{i: 9 for i in range(241, 244)},
        # Cropland stable/gain/loss (244-249)
        **{i: 5 for i in range(244, 250)},
        # Built-up stable/gain/loss (250-253)
        **{i: 7 for i in range(250, 254)},
        # Ocean (254)
        254: 1,
        # No data (255)
        255: 0,
    }
    NEW_GLAD_CLASS_MAPPING = {}
    for key, value in GLAD_TO_CLASSES.items():
        if value is not None:
            NEW_GLAD_CLASS_MAPPING[key] = ESRI_LAND_COVERclass_conversion_dict[value]
    return NEW_GLAD_CLASS_MAPPING


class BaseLandCover(BaseEnvironmentalMetric):
    def __init__(
        self,
        collections: List[str],
        band_name: str,
        name: str,
        class_conversion_dict: Dict[int, str],
        sources: List[str],
        resolution: int = 0.1,
        max_workers: int = 10,
        show_progress: bool = True,
    ) -> None:
        super().__init__()
        self.collections = collections
        self.band_name = band_name
        self.name = name
        self.class_conversion_dict = class_conversion_dict
        self.sources = sources
        self.resolution = resolution
        self.max_workers = max_workers
        self.show_progress = show_progress

    def get_items(
        self, start_date: str, end_date: str, polygon: dict, polygon_crs: str
    ) -> List[Item]:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        return get_planetary_computer_items(
            collections=self.collections,
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
        )

    def load_xarray(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str,
    ) -> xr.Dataset:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        logger.debug(f"Loading {self.collections} data at {self.resolution}m resolution")
        items = self.get_items(
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
            polygon_crs=polygon_crs,
        )

        if not items:
            raise ValueError(
                f"No {self.name} items found for the given date range and polygon"
            )

        signed_items = [planetary_computer.sign(item) for item in items]
        thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)

        return odc.stac.load(
            signed_items,
            bands=[self.band_name],
            resolution=self.resolution,
            pool=thread_pool,
            geopolygon=polygon,
            progress=tqdm if self.show_progress else None,
        )

    def getclass_conversion_dict(self) -> Dict[int, str]:
        return self.class_conversion_dict

    def get_xarray_with_class_names(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str,
    ):
        ds = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
            polygon_crs=polygon_crs,
        )
        return ds.assign_coords(
            **{self.band_name: ds[self.band_name].map(self.class_conversion_dict)}
        )

    def get_land_use_class_percentages(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str,
        all_touched: bool = True,
    ) -> pd.DataFrame:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        images = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
            polygon_crs=polygon_crs,
        )

        crs = images.coords["spatial_ref"].values.item()
        clipped_data = images.rio.write_crs(crs).rio.clip(
            [polygon], polygon_crs, all_touched=all_touched
        )
        clipped_data = clipped_data.where(clipped_data[self.band_name] != 0)

        clipped_data_df = clipped_data.to_dataframe()
        clipped_data_df[self.band_name] = clipped_data_df[self.band_name].map(
            self.class_conversion_dict
        )
        grouped = clipped_data_df.groupby("time")
        value_counts = grouped[self.band_name].value_counts()
        total_counts = grouped[self.band_name].count()

        percentage = (value_counts / total_counts).unstack(level=1)
        return round(percentage * 100, 2)

    def get_data(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str,
        all_touched: bool = True,
    ) -> Dict:
        """Get land use class percentages for a given geometry"""
        return self.get_land_use_class_percentages(
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
            polygon_crs=polygon_crs,
            all_touched=all_touched,
        )


class EsaLandCover(BaseLandCover):
    def __init__(self, use_esri_classes: bool = False) -> None:
        super().__init__(
            collections=["esa-cci-lc"],
            sources=[
                "https://planetarycomputer.microsoft.com/dataset/esa-cci-lc",
                "https://doi.org/10.24381/cds.006f2c9a",
            ],
            name="ESA Climate Change Initiative (CCI) Land Cover",
            band_name="lccs_class",
            class_conversion_dict=ESA_LAND_COVERclass_conversion_dict
            if not use_esri_classes
            else map_esa_to_esri_classes(),
        )


class EsriLandCover(BaseLandCover):
    def __init__(self) -> None:
        super().__init__(
            collections=["io-lulc-annual-v02"],
            sources=[
                "https://planetarycomputer.microsoft.com/dataset/io-lulc-annual-v02#Example-Notebook"
            ],
            name="Esri Land Use",
            band_name="data",
            class_conversion_dict=ESRI_LAND_COVERclass_conversion_dict,
        )


class OpenLandMapLandCover(BaseLandCover):
    def __init__(self, use_esri_classes: bool = False) -> None:
        super().__init__(
            collections=None,
            sources=["https://glad.umd.edu/dataset/GLCLUC"],
            name="GLAD Land Use/Cover",
            band_name="data",
            class_conversion_dict=map_glad_to_esri_classes()
            if use_esri_classes
            else GLAD_LAND_COVERclass_conversion_dict,
        )

    def load_xarray(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str = "EPSG:4326",
    ) -> List[Item]:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        """Override get_items to use GLAD land cover data instead of Planetary Computer"""
        # Convert dates to years
        start_year = str(pd.to_datetime(start_date).year)
        end_year = str(pd.to_datetime(end_date).year)

        # Get available years within range
        available_years = [
            y for y in OPENLANDMAP_LC.keys() if start_year <= y <= end_year
        ]

        if not available_years:
            raise ValueError(
                f"No GLAD data available between {start_year} and {end_year}"
            )

        # Load and merge data for all available years
        data_arrays = []
        
        minx, miny, maxx, maxy = gpd.GeoDataFrame([polygon], columns=["geometry"]).set_geometry("geometry").bounds.iloc[
            0
        ]
        # somehow faster than  
        for year in available_years:
            url = OPENLANDMAP_LC[year]
            da = rioxarray.open_rasterio(url)
            da = da.assign_coords(time=pd.Timestamp(f"{year}-01-01"))
            da = da.rio.clip_box(
                minx=minx, miny=miny, maxx=maxx, maxy=maxy, crs=polygon_crs
            )
            data_arrays.append(da)
        ds = xr.concat(data_arrays, dim="time")
        ds = ds.squeeze()
        return ds

    def get_land_use_class_percentages(
        self,
        start_date: str,
        end_date: str,
        polygon: dict,
        polygon_crs: str,
        all_touched: bool = True,
    ) -> pd.DataFrame:
        polygon = self._preprocess_geometry(polygon, source_crs=polygon_crs)
        ds = self.load_xarray(
            start_date=start_date,
            end_date=end_date,
            polygon=polygon,
            polygon_crs=polygon_crs,
        )
        clipped_data = ds.rio.clip(
            [polygon], polygon_crs, all_touched=all_touched
        )
        clipped_data_df = clipped_data.to_dataframe("class").reset_index()
        clipped_data_df = clipped_data_df[clipped_data_df["class"] != 0]
        clipped_data_df["class"] = clipped_data_df["class"].map(
            self.class_conversion_dict
        )
        grouped = clipped_data_df.groupby("time")
        value_counts = grouped["class"].value_counts()
        total_counts = grouped["class"].count()

        percentage = (value_counts / total_counts).unstack(level=1)
        percentage = percentage.fillna(0)
        return round(percentage * 100, 2)
