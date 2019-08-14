import time
import propheticus

if __name__ == '__main__':
    print()
    start_time = time.time()
    propheticus.launch()

    import propheticus.shared
    propheticus.shared.Utils.printTimeLogMessage('Process', start_time)

