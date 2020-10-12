class DictKey(str):
    def __new__(cls, s, *args, label=None, **kwargs):
        dict_key_inst = super().__new__(cls, s)
        dict_key_inst.__dict__.update(kwargs)
        dict_key_inst.label = label

        return dict_key_inst


if __name__ == '__main__':
    wp_cnt_key = DictKey("wp_cnt", static=True)
    print(wp_cnt_key)
    print(wp_cnt_key.static)
