


def beauty_var_log(header: str, o: any) -> None:
    print(f"=================[ {header} ]=================")

    if isinstance(o, dict):
        for key, value in o.items():
            print(f"{key}: {value}")
    elif isinstance(o, list):
        for i, item in enumerate(o):
            print(f"[{i}] {item}")
    else:
        print(type(o))
        print(o)

    print("---------------------------------------------------")
