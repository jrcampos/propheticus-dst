
import os
import sys
import importlib
import importlib.util
import shutil
import sys

def launch():
    print(f'Python {sys.version}')

    min_major_version = 3
    min_minor_version = 6
    if sys.version_info[0] < min_major_version or sys.version_info[1] < min_minor_version:
        exit(f'\nPropheticus requires at least Python {min_major_version}.{min_minor_version}\n')

    RequiredPackages = [
        'pypyodbc',
        'numpy',
        'sklearn',
        'openpyxl',
        'xlrd',
        'matplotlib',
        'imblearn',
        'statsmodels',
        'pyclustering',
        'seaborn',
        'joblib',
        'pandas',
        'graphviz',
        'tensorflow',
        'matplotlib_venn',
        'pydotplus',
        'pexpect',
        'xgboost',
        'keras',
        'art'  # NOTE: adversarial-robustness-toolbox
    ]

    MissingModules = [required_module for required_module in RequiredPackages if not importlib.util.find_spec(required_module)]
    if MissingModules:
        exit('The following modules are required for Propheticus: \n' + '\n'.join(MissingModules) + '\n')

    Configs = {'framework_instance': None}

    import propheticus
    # if os.path.exists(propheticus.Config.framework_temp_path):
    #     shutil.rmtree(propheticus.Config.framework_temp_path)

    Instances = next(os.walk(propheticus.Config.framework_instances_path))[1]
    if len(Instances) > 1:

        breadcrumb = ' >> Initialize'
        print(''.join(['-'] * (len(breadcrumb) + 1)))
        print(breadcrumb)
        print(''.join(['-'] * (len(breadcrumb) + 1)) + '\n')

        print(' ---> Select Instance')
        for index, instance in enumerate(Instances):
            print(f' -----> {index + 1}  - {instance}')
        print('')

        char_split = 140
        message = 'ERROR: Invalid selection, please try again'
        spacer = "".join(['!' * char_split])
        spacing = "".join([' ' * int(((char_split - len(message)) / 2))])
        message = (spacing + message + spacing)[:char_split - 1]

        while True:
            print('INPUT: Select an option:')
            choice = input(" >>  ")
            if not choice.isdigit() or int(choice) < 1 or int(choice) > len(Instances):
                print(f'\n{spacer}\n {message} \n{spacer}\n')
            else:
                break

        Configs['framework_instance'] = Instances[int(choice) - 1]
    else:
        Configs['framework_instance'] = Instances[0]

    print('\n' * 2)
    print('&' * 140)
    print('&' * 140)
    print('&' * 140)
    print('\n' * 2)
    # os.system('cls')

    propheticus.Config.defineDependentPaths(Configs['framework_instance'])

    if os.path.isfile(os.path.join(propheticus.Config.framework_selected_instance_path, 'InstanceGUI.py')):
        sys.path.append(propheticus.Config.framework_selected_instance_path)
        from InstanceGUI import InstanceGUI as GUI
    else:
        import propheticus.core.GUI as GUI

    oGUI = GUI()
    oGUI.start()

    del oGUI
