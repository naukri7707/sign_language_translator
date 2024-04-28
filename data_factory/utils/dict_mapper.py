from typing import TypeVar, cast

T = TypeVar("T", dict, list, tuple)


def round_float_attribute(data: T, *, decimal_places: int = 2) -> T:
    return cast(T, __round_float_values(data, decimal_places))


def __round_float_values(data, decimal_places: int = 2):
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            if isinstance(value, float):
                new_dict[key] = round(value, decimal_places)
            elif isinstance(value, (dict, list, tuple)):
                new_dict[key] = __round_float_values(value)
            else:
                new_dict[key] = value
        return new_dict
    elif isinstance(data, (list, tuple)):
        new_list = []
        for value in data:
            if isinstance(value, float):
                new_list.append(round(value, decimal_places))
            elif isinstance(value, (dict, list, tuple)):
                new_list.append(__round_float_values(value))
            else:
                new_list.append(value)
        return new_list if isinstance(data, list) else tuple(new_list)
    else:
        return data
