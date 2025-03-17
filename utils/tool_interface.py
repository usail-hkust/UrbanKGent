from shapely.geometry import Point, LineString, Polygon
from geopy.geocoders import Nominatim
from shapely.ops import transform
from geopy.distance import geodesic
import geohash2
from shapely import wkt
from shapely.geometry import Point, Polygon
"""

LLMs 进行 geospatial reasoning 需要调用的工具集

1 计算地理实体中心点之间的距离
2 生成地理实体的地理哈希值
3 判断点是否在多边形内
4 判断点是否在线上
5 判断线是否与多边形相交
6 判断线是否在多边形内
7 多边形是否在多边形内
8 多边形是否与多边形重叠

"""


def shape_geometry_transfer(input_geometry):

    geometry = wkt.loads(input_geometry)

    return geometry
def get_centroid(geometry):
    if isinstance(geometry, Point):
        return geometry
    elif isinstance(geometry, LineString) or isinstance(geometry, Polygon):
        return geometry.centroid
    else:
        print("Unsupported geometry type")
        return None

def geohash_code(input_geometry):

    geometry = shape_geometry_transfer(input_geometry)

    # linstring和polygon为centroid的geohash编码

    if isinstance(geometry, Point):
        # For Point, use its coordinates
        return geohash2.encode(geometry.y, geometry.x, precision=8)
    elif isinstance(geometry, Polygon) or isinstance(geometry, LineString):
        # For Polygon or LineString, use the centroid coordinates
        centroid = geometry.centroid
        return geohash2.encode(centroid.y, centroid.x, precision=8)
    else:
        print("Unsupported geometry type")
        return None


def distance(geometry_1, geometry_2):
    geom1 = shape_geometry_transfer(geometry_1)
    geom2 = shape_geometry_transfer(geometry_2)


    centroid1 = get_centroid(geom1)
    centroid2 = get_centroid(geom2)

    # 获取两个centroid的坐标
    coords1 = (centroid1.y, centroid1.x)  # (纬度, 经度)
    coords2 = (centroid2.y, centroid2.x)

    # 使用Geopy计算地理距离
    distance = geodesic(coords1, coords2).kilometers

    # distance/km
    return distance


def point_belong_polygon(geometry_point, geometry_polygon):

    point = shape_geometry_transfer(geometry_point)
    polygon = shape_geometry_transfer(geometry_polygon)

    if isinstance(point, Point) and isinstance(polygon, Polygon):

        # True or False/bool
        return point.within(polygon)
    else:
        print("Unsupported geometry type")
        return None




def point_intersects_linestring(geometry_point, geometry_linestring):
    point = shape_geometry_transfer(geometry_point)
    linestring = shape_geometry_transfer(geometry_linestring)


    if isinstance(point, Point) and isinstance(linestring, LineString):
        # True or False/bool
        return point.intersects(linestring)
    else:
        print("Unsupported geometry type")
        return None


def linestring_intersect_ploygon(geometry_linestring, geometry_polygon):
    linestring = shape_geometry_transfer(geometry_linestring)
    polygon = shape_geometry_transfer(geometry_polygon)


    if isinstance(linestring, LineString) and isinstance(polygon, Polygon):
        # True or False/bool
        return linestring.intersects(polygon)
    else:
        print("Unsupported geometry type")
        return None



def linestring_belong_ploygon(geometry_linestring, geometry_polygon):

    linestring = shape_geometry_transfer(geometry_linestring)
    polygon = shape_geometry_transfer(geometry_polygon)


    if isinstance(linestring, LineString) and isinstance(polygon, Polygon):
        # True or False/bool
        return linestring.within(polygon)
    else:
        print("Unsupported geometry type")
        return None


def polygon_intersect_ploygon(geometry_1, geometry_2):

    polygon1 = shape_geometry_transfer(geometry_1)
    polygon2 = shape_geometry_transfer(geometry_2)

    if isinstance(polygon1, Polygon) and isinstance(polygon2, Polygon):
        return polygon1.intersects(polygon2)
    else:
        print("Unsupported geometry type")
        return None

def polygon_belong_ploygon(geometry_1, geometry_2):

    polygon1 = shape_geometry_transfer(geometry_1)
    polygon2 = shape_geometry_transfer(geometry_2)

    if isinstance(polygon1, Polygon) and isinstance(polygon2, Polygon):
        return polygon1.within(polygon2)
    else:
        print("Unsupported geometry type")
        return None

