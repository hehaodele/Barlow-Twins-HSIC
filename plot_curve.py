from header import *
from local import *
import pandas as pd

fn_list = os.listdir(result_path)
print(fn_list)

for fn in fn_list:
    df = pd.read_csv(f'{result_path}/{fn}')
    plt.plot(df['epoch'], df['test_acc@1'], label=fn)
plt.legend()
plt.grid()
plt.show()