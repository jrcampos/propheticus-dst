import functools
import os
import sys
import importlib

import propheticus

# sys.path.append(propheticus.Config.framework_selected_instance_path)

def custom_hook():
    def decorator_func(func):
        @functools.wraps(func)
        def wrapper_decorator_func(*args, **kwargs):
            # NOTE: the hook functions should have the exact same signature (param names included)!

            context_classname = type(args[0]).__name__
            target_class = f'Instance{context_classname}'
            if not os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, f'{target_class}.py')):
                value = func(*args, **kwargs)
                return value

            loaded_module = importlib.import_module(f'{target_class}')
            CustomClass = getattr(loaded_module, target_class)
            # Do something before

            target_function_name = f'precall_{func.__name__}'
            if hasattr(CustomClass, target_function_name):
                getattr(CustomClass, target_function_name)(*args, **kwargs)

            value = func(*args, **kwargs)

            target_function_name = f'postcall_{func.__name__}'
            if hasattr(CustomClass, target_function_name):
                _value = value
                _value = getattr(CustomClass, target_function_name)(value, *args, **kwargs)
                if _value is None:
                    propheticus.shared.Utils.printFatalMessage(f'Custom hook {target_function_name} must return values!')
                value = _value

            # Do something after
            return value

        return wrapper_decorator_func
    return decorator_func