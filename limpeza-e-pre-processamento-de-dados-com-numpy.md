###### by Anderson Paiva [Linkedin](https://www.linkedin.com/in/anderson-paiva/)/[Github](https://github.com/And3rson-Paiva)

# Limpeza e Pré-Processamento de Dados com Numpy

<div align="center">
    <img width=600 title="titulo da imagem" src="Data%20processing-bro.png"/>
</div>


# Estudo de caso

Para  este  projeto trabalhei  com  dados  reais  disponíveis  publicamente  no 
link: https://www.openintro.org/data/index.php?data=loans_full_schema

Imagine que em um determinado projeto de Ciência de Dados que você receba um dataset extremamente complicado, 
contendo dados com muitas strings, caracteres especiais, problemas de encoding, datas mal formatadas, 
números e textos na mesma  coluna, url’s contendo Ids importantes para análise, valores ausentes, 
coluna que contém informação que deveria estar distribuída em três ou mais colunas. 

<div align="center">
    <img width=600 title="titulo da imagem" src="nuvem_de_palavras1.jpg"/>
</div>

E como se não bastasse tudo isso, parte dos dados necessários para análise está em outro dataset,
que deve ser combinado com o primeiro.

<div align="center">
    <img width=400 title="titulo da imagem" src="4966816c313ceed673d885ba61bacea1.jpg"/>
</div>

É exatamente este cenário que estou reproduzindo nesse projeto.
A partir de dados complexos e com diversos problemas, iremos fazer um extenso trabalho de 
limpeza e pré-processamento. E tudo isso usando apenas o **NumPy**, poderoso pacote 
para computação e processamento de dados.


O conjunto de dados que vou trabalhar representa milhares de empréstimos feitos  por  meio  da plataforma  Lending  Club, que é uma plataforma que permite que indivíduos emprestem para outros indivíduos.

Claro, nem todos os empréstimos são iguais. Alguém que fornece um baixo risco e que provavelmente vai pagar um empréstimo terá mais facilidade em obter um empréstimo com uma taxa de juros baixa do que alguém que parece ser mais arriscado.

E para as pessoas com alto risco de não pagar o empréstimo? 

Essas pessoas podem nem receber uma oferta de empréstimo, ou podem não aceitar uma oferta de empréstimo devido a uma alta taxa de juros. 
**É importante ter em mente essa última parte, pois esse conjunto de dados representa  apenas  empréstimos efetivamente  feitos,  ou  seja,  não  confundir  esses  dados  com pedidos de empréstimo!**

Além  disso  usei  um  dataset  com  cotação  do  dólar  em  relação  ao  Euro.  
Extraí uma pequena amostra de dados do site: https://finance.yahoo.com

**Pacote Python utilizado nesse projeto**
<div align="center">
    <img width=400 title="titulo da imagem" src="índice.png"/>
</div>


```python
# imports do projeto
from platform import python_version
import warnings
import numpy as np

# Ignora avisos de warnings
warnings.filterwarnings('ignore')
```


```python
# versão dos pacotes instalados neste notebook
%reload_ext watermark
%watermark  -a "Anderson Paiva" --iversions
```

    Author: Anderson Paiva
    
    numpy: 1.23.4
    



```python
print(f"Versão Python {python_version()}")
```

    Versão Python 3.9.2



```python
# Configuração de impressão do Numpy
np.set_printoptions(suppress = True, linewidth = 200, precision = 2)
```

# Carregando o dataset


```python
dados = np.genfromtxt(
    'dataset1.csv',
    delimiter = ';',
    skip_header = 1,
    autostrip = True,
    encoding = 'cp1252'
)
```


```python
# tipo de dado
type(dados)
```




    numpy.ndarray




```python
# formato do dados
dados.shape
```




    (10000, 14)




```python
# Visualização dos dados.
# Já encontramos um erro, o nan no dataset (valores ausentes), esse valor 
# foi gerado pelo próprio numpy que não reconheceu alguns caracteres especiais no conjunto de dados 
# e a forma como o numpy carrega dados do tipo string e númerico.
# Ou seja, o problema não esta nos dados em si, mas sim na forma 
# como o numpy carregou o dataset.
dados.view()
```




    array([[48010226.  ,         nan,    35000.  , ...,         nan,         nan,     9452.96],
           [57693261.  ,         nan,    30000.  , ...,         nan,         nan,     4679.7 ],
           [59432726.  ,         nan,    15000.  , ...,         nan,         nan,     1969.83],
           ...,
           [50415990.  ,         nan,    10000.  , ...,         nan,         nan,     2185.64],
           [46154151.  ,         nan,         nan, ...,         nan,         nan,     3199.4 ],
           [66055249.  ,         nan,    10000.  , ...,         nan,         nan,      301.9 ]])



# Verificando valores ausentes


```python
# total de valores ausentes
np.isnan(dados).sum()
```




    88005




```python
# retornando o maior valor no conjunto de dados ignorando valores nan
# usei esse valor arbitrário para preencher os valores ausentes na carga de dados numéricos
# mais a frente esses dados seram tratados como ausentes.
valor_coringa = np.nanmax(dados) + 1
print(valor_coringa)
```

    68616520.0



```python
# calculo da média ignorando valor nan por coluna
# será usado para separar variáveis numéricas e variáveis do tipo string
media_ignorando_nan = np.nanmean(dados, axis = 0)
print(media_ignorando_nan)
```

    [54015809.19         nan    15273.46         nan    15311.04         nan       16.62      440.92         nan         nan         nan         nan         nan     3143.85]



```python
# colunas do tipo string com valores ausentes
colunas_string = np.argwhere(np.isnan(media_ignorando_nan)).squeeze()
print(colunas_string)
```

    [ 1  3  5  8  9 10 11 12]



```python
# colunas numéricas
colunas_numericas = np.argwhere(np.isnan(media_ignorando_nan) == False).squeeze()
print(colunas_numericas)
```

    [ 0  2  4  6  7 13]


### Importando novamente os dados, agora de forma separada as colunas do tipo string e as colunas do tipo numéricas


```python
arr_strings = np.genfromtxt(
    'dataset1.csv',
    delimiter = ';',
    skip_header = 1,
    autostrip = True,
    usecols = colunas_string,
    dtype = str,
    encoding = 'cp1252'
)
```


```python
print(arr_strings)
```

    [['May-15' 'Current' '36 months' ... 'Verified' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226' 'CA']
     ['' 'Current' '36 months' ... 'Source Verified' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261' 'NY']
     ['Sep-15' 'Current' '36 months' ... 'Verified' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726' 'PA']
     ...
     ['Jun-15' 'Current' '36 months' ... 'Source Verified' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990' 'CA']
     ['Apr-15' 'Current' '36 months' ... 'Source Verified' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151' 'OH']
     ['Dec-15' 'Current' '36 months' ... '' 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249' 'IL']]



```python
arr_numeric = np.genfromtxt(
    'dataset1.csv',
    delimiter = ';',
    skip_header = 1,
    autostrip = True,
    usecols = colunas_numericas,
    filling_values = valor_coringa,
    dtype = int,
    encoding = 'cp1252'
)
```


```python
print(arr_numeric)
```

    [[48010226 68616520 68616520 68616520 68616520 68616520]
     [57693261 68616520 68616520 68616520 68616520 68616520]
     [59432726 68616520 68616520 68616520 68616520 68616520]
     ...
     [50415990 68616520 68616520 68616520 68616520 68616520]
     [46154151 68616520 68616520 68616520 68616520 68616520]
     [66055249 68616520 68616520 68616520 68616520 68616520]]


#### Agora vou extrair o nome das colunas


```python
arr_nomes_colunas = np.genfromtxt(
    'dataset1.csv',
    delimiter = ';',
    autostrip = True,
    skip_footer = dados.shape[0],
    dtype = str,
    encoding = 'cp1252'
)
```


```python
print(arr_nomes_colunas)
```

    ['id' 'issue_d' 'loan_amnt' 'loan_status' 'funded_amnt' 'term' 'int_rate' 'installment' 'grade' 'sub_grade' 'verification_status' 'url' 'addr_state' 'total_pymnt']



```python
# Separando o cabeçalho de colunas string e numéricas
header_string, header_numeric = arr_nomes_colunas[colunas_string], arr_nomes_colunas[colunas_numericas]
```


```python
print(header_string)
```

    ['issue_d' 'loan_status' 'term' 'grade' 'sub_grade' 'verification_status' 'url' 'addr_state']



```python
print(header_numeric)
```

    ['id' 'loan_amnt' 'funded_amnt' 'int_rate' 'installment' 'total_pymnt']


# Função Checkpoint


```python
# resolvi criar uma função para gerar um checkpoint e salvar os dados gerados até o momento
def checkpoint(file_name, checkpoint_header, checkpoint_data):
    np.savez(file_name, header = checkpoint_header, data = checkpoint_data)
    checkpoint_variable = np.load(file_name + ".npz")
    return (checkpoint_variable)
```


```python
check_point_inicial = checkpoint("Checkpoint_inicial", header_string, arr_strings)
```


```python
# checkpoint criado
check_point_inicial['data']
```




    array([['May-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'CA'],
           ['', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261', 'NY'],
           ['Sep-15', 'Current', '36 months', ..., 'Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', 'PA'],
           ...,
           ['Jun-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990', 'CA'],
           ['Apr-15', 'Current', '36 months', ..., 'Source Verified', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'OH'],
           ['Dec-15', 'Current', '36 months', ..., '', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249', 'IL']], dtype='<U69')




```python
# comparando se o checkpoint é igual ao array de strings
np.array_equal(check_point_inicial['data'], arr_strings)
```




    True



# Manipulando as colunas do tipo string


```python
# 8 colunas categoricas
header_string
```




    array(['issue_d', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# vou ajustar o nome da coluna para facilitar a identificação
header_string[0] = "issue_date"
```


```python
# nome da coluna modificado no array
header_string
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')



# Pré-Processamento da variável issue_date com Label Encoding


```python
# extrai os valores únicos da variável
np.unique(arr_strings[:, 0])
```




    array(['', 'Apr-15', 'Aug-15', 'Dec-15', 'Feb-15', 'Jan-15', 'Jul-15', 'Jun-15', 'Mar-15', 'May-15', 'Nov-15', 'Oct-15', 'Sep-15'], dtype='<U69')




```python
# removi o sufixo -15 e fiz a conversão em um array de string para posteriormente
# aplicar a técnica de label encoding
arr_strings[:, 0] = np.chararray.strip(arr_strings[:, 0], "-15")
```


```python
# visualização dos dados únicos da variável já modificada
np.unique(arr_strings[:, 0])
```




    array(['', 'Apr', 'Aug', 'Dec', 'Feb', 'Jan', 'Jul', 'Jun', 'Mar', 'May', 'Nov', 'Oct', 'Sep'], dtype='<U69')




```python
# criei um array de meses considerando o valor vazio para o que estiver em branco
meses = np.array(['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
```


```python
# loop para converter o nome dos meses em valores numéricos
# correspondentes, esse processo é chamado de label encoding
for i in range(13):
    arr_strings[:, 0] = np.where(arr_strings[:, 0] == meses[i], i, arr_strings[:, 0])
```


```python
# verificando os valores após a conversão
np.unique(arr_strings[:, 0])
```




    array(['0', '1', '10', '11', '12', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U69')



# Pré-Processamento da variável loan_status com Binarização


```python
header_string
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# extraindo valores unicos
np.unique(arr_strings[:, 1])
```




    array(['', 'Charged Off', 'Current', 'Default', 'Fully Paid', 'In Grace Period', 'Issued', 'Late (16-30 days)', 'Late (31-120 days)'], dtype='<U69')




```python
# número de elementos/categorias/status
np.unique(arr_strings[:, 1]).size
```




    9




```python
# no caso resolvi utilizar somente 3 elementos disponíveis, pois de acordo com a nossa are de "negócio" não seria
# necessário utilizar todos os elementos disponíveis para gerar o resultado esperado.
# Criei então um array com os 3 elementos.
status_bad = np.array(['', 'Charged Off', 'Default', 'Late (16-30 days)'])
```


```python
# faço a checagem dos valores da variável e comparo com o valor da variável anterior, convertendo a variável para
# valor binário, esse processo é chamado de Binarização.
arr_strings[:, 1] = np.where(np.isin(arr_strings[:, 1], status_bad), 0, 1)
```


```python
# extraindo valores unicos da variável
# 0 (zero) será a classe negativa e 1 (um) será a classe positiva
np.unique(arr_strings[:, 1])
```




    array(['0', '1'], dtype='<U69')



# Pré-Processamento da variável term com limpeza de string


```python
header_string
```




    array(['issue_date', 'loan_status', 'term', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# extraindo valores unicos
np.unique(arr_strings[:, 2])
```




    array(['', '36 months', '60 months'], dtype='<U69')




```python
# removendo a palavra months ()
arr_strings[:, 2] = np.chararray.strip(arr_strings[:, 2], " months")
arr_strings[:, 2]
```




    array(['36', '36', '36', ..., '36', '36', '36'], dtype='<U69')




```python
# mudando o titulo da variável
header_string[2] = "term_months"
```


```python
# Subistitui os valores ausentes pelo maior valor, que no meu caso é 60.
# Escolhi esse valor (60) dado o fato de não ter a informação correta da "area de negócios".
arr_strings[:, 2] = np.where(arr_strings[:, 2] == "", '60', arr_strings[:, 2])
```


```python
# extraindo valores unicos da variável
np.unique(arr_strings[:, 2])
```




    array(['36', '60'], dtype='<U69')



### Pré-Processamento das variáveis grade e sub_grade com dicionário (Tipo de label Encoding)


```python
header_string
```




    array(['issue_date', 'loan_status', 'term_months', 'grade', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai os valores unicos da variável
np.unique(arr_strings[:, 3])
```




    array(['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
# Extrai os valores unicos da variável
np.unique(arr_strings[:, 4])
```




    array(['', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
           'G2', 'G3', 'G4', 'G5'], dtype='<U69')



#### Vou ajustar a variável sub_grade por um único motivo, ela tem mais detalhes e parece ser melhor para o processo


```python
np.unique(arr_strings[:, 3])
```




    array(['', 'A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
np.unique(arr_strings[:, 3])[1:]
```




    array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype='<U69')




```python
# loop para ajuste da variável sub_grade
for i in np.unique(arr_strings[:, 3])[1:]:
    arr_strings[:, 4] = np.where((arr_strings[:, 4] == '') & (arr_strings[:, 3] == i), i + '5', arr_strings[:, 4])
```


```python
# retorna a categoria e sua respectiva contagem
np.unique(arr_strings[:, 4], return_counts = True)
```




    (array(['', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1',
            'G2', 'G3', 'G4', 'G5'], dtype='<U69'),
     array([  9, 285, 278, 239, 323, 592, 509, 517, 530, 553, 633, 629, 567, 586, 564, 577, 391, 267, 250, 255, 288, 235, 162, 171, 139, 160,  94,  52,  34,  43,  24,  19,  10,   3,   7,   5]))




```python
# Subistitui valores ausentes por uma nova categoria
arr_strings[:, 4] = np.where(arr_strings[:, 4] == '', 'H1', arr_strings[:, 4])
```


```python
# Extrai os valores unicos da variável
np.unique(arr_strings[:, 4])
```




    array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2',
           'G3', 'G4', 'G5', 'H1'], dtype='<U69')



#### Vou remover a variável grade


```python
# essa variável não é mais necessária no processo já que escolhi usar a variável sub_grade
arr_strings = np.delete(arr_strings, 3, axis = 1)
```


```python
# após deletar a variável grade, a variável sub_grade passa a ocupar o indice da que foi deletada
arr_strings[:, 3]
```




    array(['C3', 'A5', 'B5', ..., 'A5', 'D2', 'A4'], dtype='<U69')




```python
# Nesse caso removi o nome da coluna do array de nome de colunas
header_string = np.delete(header_string, 3)
```


```python
# Nova variávrel na coluna de índice 3
header_string[3]
```




    'sub_grade'



#### Por fim, vou converter a variável sub_grade para sua representação numérica


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:, 3])
```




    array(['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'G1', 'G2',
           'G3', 'G4', 'G5', 'H1'], dtype='<U69')




```python
# Cria uma lista de chaves
keys = list(np.unique(arr_strings[:, 3]))
keys[0]
```




    'A1'




```python
# Cria uma lista de valores
values = list(range(1, np.unique(arr_strings[:, 3]).shape[0] + 1))
values[0]
```




    1




```python
# Criando o dicionário
dict_sub_grade = dict(zip(keys, values))
dict_sub_grade
```




    {'A1': 1,
     'A2': 2,
     'A3': 3,
     'A4': 4,
     'A5': 5,
     'B1': 6,
     'B2': 7,
     'B3': 8,
     'B4': 9,
     'B5': 10,
     'C1': 11,
     'C2': 12,
     'C3': 13,
     'C4': 14,
     'C5': 15,
     'D1': 16,
     'D2': 17,
     'D3': 18,
     'D4': 19,
     'D5': 20,
     'E1': 21,
     'E2': 22,
     'E3': 23,
     'E4': 24,
     'E5': 25,
     'F1': 26,
     'F2': 27,
     'F3': 28,
     'F4': 29,
     'F5': 30,
     'G1': 31,
     'G2': 32,
     'G3': 33,
     'G4': 34,
     'G5': 35,
     'H1': 36}




```python
# loop para substituir a string com as categoria pela representação numérica (frequência)
for i in np.unique(arr_strings[:, 3]):
    arr_strings[:, 3] = np.where(arr_strings[:, 3] == i, dict_sub_grade[i], arr_strings[:, 3])
```


```python
# Extrai o valor único das variáveis
np.unique(arr_strings[:, 3])
```




    array(['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '4', '5', '6',
           '7', '8', '9'], dtype='<U69')



# Pré-Processamento da variável verification status com binarização


```python
# lista com os nomes das variáveis
header_string
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'url', 'addr_state'], dtype='<U19')




```python
# Extrai valores únicos das variáveis
np.unique(arr_strings[:, 4])
```




    array(['', 'Not Verified', 'Source Verified', 'Verified'], dtype='<U69')




```python
# Utilizei a binarização nessa variável
# 0 (zero) é a categoria negativa Not Verified e 1 (um) a categoria positiva Verified/Source Verified
arr_strings[:, 4] = np.where((arr_strings[:, 4] == '') | (arr_strings[:, 4] == 'Not Verified'), 0, 1)
```


```python
# Extrai os valortes únicos da variável
np.unique(arr_strings[:, 4])
```




    array(['0', '1'], dtype='<U69')



# Pré-Processamento da variável url com a extração de ID


```python
# visualiza a amostra dos dados
arr_strings[:, 5]
```




    array(['https://www.lendingclub.com/browse/loanDetail.action?loan_id=48010226', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=57693261',
           'https://www.lendingclub.com/browse/loanDetail.action?loan_id=59432726', ..., 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=50415990',
           'https://www.lendingclub.com/browse/loanDetail.action?loan_id=46154151', 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=66055249'], dtype='<U69')




```python
# Extrai o ID ao final de cada url
np.chararray.strip(arr_strings[:, 5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')
```




    chararray(['48010226', '57693261', '59432726', ..., '50415990', '46154151', '66055249'], dtype='<U69')




```python
# Substitui a url pelo valor do ID na url
arr_strings[:, 5] = np.chararray.strip(arr_strings[:, 5], 'https://www.lendingclub.com/browse/loanDetail.action?loan_id=')
```


```python
# Convertemos o tipo para int32
arr_strings[:, 5].astype(dtype = np.int32)
```




    array([48010226, 57693261, 59432726, ..., 50415990, 46154151, 66055249], dtype=int32)




```python
# Esse ID esta presente na primeira coluna de dados, vou converter para int32 e comparar
arr_numeric[:, 0].astype(dtype = np.int32)
```




    array([48010226, 57693261, 59432726, ..., 50415990, 46154151, 66055249], dtype=int32)




```python
# Verificando se os IDs são iguais, e sim, as duas variáveis são identicas.
# O ID extraido da url é igual ao ID que temos no connjunto de dados.
np.array_equal(arr_numeric[:, 0].astype(dtype = np.int32), arr_strings[:, 5].astype(dtype = np.int32))
```




    True




```python
# Remove do array de dados
arr_strings = np.delete(arr_strings, 5, axis = 1)
```


```python
# Remove do array de nome de coluna
header_string = np.delete(header_string, 5)
```


```python
# Nova coluna no indice 5
arr_strings[:, 5]
```




    array(['CA', 'NY', 'PA', ..., 'CA', 'OH', 'IL'], dtype='<U69')




```python
# Nova lista de colunas
header_string
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'addr_state'], dtype='<U19')




```python
# Coluna ID
arr_numeric[:, 0]
```




    array([48010226, 57693261, 59432726, ..., 50415990, 46154151, 66055249])




```python
# Coluna ID agora faz parte do array de numéricos
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')



# Pré-Processamento da variável address com Categorização


```python
header_string
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'addr_state'], dtype='<U19')




```python
# Ajustando o nome da coluna address
header_string[5] = "state_address"
```


```python
# Extrai nomes e contagens
states_name, states_count = np.unique(arr_strings[:, 5], return_counts = True)
```


```python
# Ordenando em ordem decrescente
states_count_sorted = np.argsort(-states_count)
```


```python
# resultado
states_name[states_count_sorted], states_count[states_count_sorted]
```




    (array(['CA', 'NY', 'TX', 'FL', '', 'IL', 'NJ', 'GA', 'PA', 'OH', 'MI', 'NC', 'VA', 'MD', 'AZ', 'WA', 'MA', 'CO', 'MO', 'MN', 'IN', 'WI', 'CT', 'TN', 'NV', 'AL', 'LA', 'OR', 'SC', 'KY', 'KS', 'OK',
            'UT', 'AR', 'MS', 'NH', 'NM', 'WV', 'HI', 'RI', 'MT', 'DE', 'DC', 'WY', 'AK', 'NE', 'SD', 'VT', 'ND', 'ME'], dtype='<U69'),
     array([1336,  777,  758,  690,  500,  389,  341,  321,  320,  312,  267,  261,  242,  222,  220,  216,  210,  201,  160,  156,  152,  148,  143,  143,  130,  119,  116,  108,  107,   84,   84,   83,
              74,   74,   61,   58,   57,   49,   44,   40,   28,   27,   27,   27,   26,   25,   24,   17,   16,   10]))




```python
# Substituindo valores ausentes por zero (0)
arr_strings[:, 5]= np.where(arr_strings[:, 5] == '', 0, arr_strings[:, 5])
```

### Separando os estados por regiões


```python
states_west = np.array(['WA', 'OR','CA','NV','ID','MT', 'WY','UT','CO', 'AZ','NM','HI','AK'])
states_south = np.array(['TX','OK','AR','LA','MS','AL','TN','KY','FL','GA','SC','NC','VA','WV','MD','DE','DC'])
states_midwest = np.array(['ND','SD','NE','KS','MN','IA','MO','WI','IL','IN','MI','OH'])
states_east = np.array(['PA','NY','NJ','CT','MA','VT','NH','ME','RI'])
```


```python
# Agora vou substituir cada estado pelo ID de sua região
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_west), 1, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_south), 2, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_midwest), 3, arr_strings[:,5])
arr_strings[:,5] = np.where(np.isin(arr_strings[:,5], states_east), 4, arr_strings[:,5])
```


```python
# Extrai os valores únicos da variável
np.unique(arr_strings[:, 5])
```




    array(['0', '1', '2', '3', '4'], dtype='<U69')



# Convertendo o Array


```python
# Vou modificar o tipo de dado do array abaixo, o array esta usando números mas com formato dos dados errado.
arr_strings
```




    array([['5', '1', '36', '13', '1', '1'],
           ['0', '1', '36', '5', '1', '4'],
           ['9', '1', '36', '10', '1', '4'],
           ...,
           ['6', '1', '36', '5', '1', '1'],
           ['4', '1', '36', '17', '1', '3'],
           ['12', '1', '36', '4', '0', '3']], dtype='<U69')




```python
# Agora o array esta no formato correto
arr_strings = arr_strings.astype(int)
```


```python
arr_strings
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  0,  3]])



# Checkpoint com variáveis do tipo string limpas e pré-processadas


```python
# Checkpoint
checkpoint_strings = checkpoint("Checkpoint-Strings", header_string, arr_strings)
```


```python
# Cabeçalho
checkpoint_strings["header"]
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'state_address'], dtype='<U19')




```python
# Os dados
checkpoint_strings["data"]
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  0,  3]])




```python
# Comparação, se o checkpoint salvo em disco esta igual ao que esta na memória do computador
np.array_equal(checkpoint_strings['data'], arr_strings)
```




    True



# Manipulando colunas numéricas


```python
# Visualizando os dados
arr_numeric
```




    array([[48010226, 68616520, 68616520, 68616520, 68616520, 68616520],
           [57693261, 68616520, 68616520, 68616520, 68616520, 68616520],
           [59432726, 68616520, 68616520, 68616520, 68616520, 68616520],
           ...,
           [50415990, 68616520, 68616520, 68616520, 68616520, 68616520],
           [46154151, 68616520, 68616520, 68616520, 68616520, 68616520],
           [66055249, 68616520, 68616520, 68616520, 68616520, 68616520]])




```python
# Nome das colunas
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
# Agora não temos mais valores ausentes, pois ao carraga os dados usamos um valor arbitrário
np.isnan(arr_numeric).sum()
```




    0




```python
# Verificando de 2 formas diferentes, se a coluna foi preenchida com o valor coringa
np.isin(arr_numeric[:, 0], valor_coringa)
```




    array([False, False, False, ..., False, False, False])




```python
np.isin(arr_numeric[:, 0], valor_coringa).sum()
```




    0



#### Agora vou criar um array de estatísticas, especificamente valor mínimo, máximo e média de cada variável. Usaremos isso no tratamento de valores ausentes (preenchidos com o valor coringa).


```python
# Criei um array com valor mínimo, média e valor máximo ignorando nan.
# Isso será usado no tratamento de valores ausentes
arr_stats = np.array([np.nanmin(dados, axis = 0), media_ignorando_nan, np.nanmax(dados, axis = 0)])
```


```python
arr_stats
```




    array([[  373332.  ,         nan,     1000.  ,         nan,     1000.  ,         nan,        6.  ,       31.42,         nan,         nan,         nan,         nan,         nan,        0.  ],
           [54015809.19,         nan,    15273.46,         nan,    15311.04,         nan,       16.62,      440.92,         nan,         nan,         nan,         nan,         nan,     3143.85],
           [68616519.  ,         nan,    35000.  ,         nan,    35000.  ,         nan,       28.99,     1372.97,         nan,         nan,         nan,         nan,         nan,    41913.62]])




```python
arr_stats[:, colunas_numericas]
```




    array([[  373332.  ,     1000.  ,     1000.  ,        6.  ,       31.42,        0.  ],
           [54015809.19,    15273.46,    15311.04,       16.62,      440.92,     3143.85],
           [68616519.  ,    35000.  ,    35000.  ,       28.99,     1372.97,    41913.62]])



## Pré-Processamento da Variável funded_amnt


```python
# Visualiza os dados
arr_numeric[:,2]
```




    array([68616520, 68616520, 68616520, ..., 68616520, 68616520, 68616520])




```python
arr_stats[0, colunas_numericas[2]]
```




    1000.0




```python
# Ajustando o conteúdo da coluna
arr_numeric[:,2] = np.where(arr_numeric[:,2] == valor_coringa, arr_stats[0, colunas_numericas[2]], arr_numeric[:,2])
```


```python
arr_numeric[:, 2]
```




    array([1000, 1000, 1000, ..., 1000, 1000, 1000])



### Pré-Processamento das Variáveis loan_amnt, int_rate, installment e total_pymnt


```python
# Nomes das colunas
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt'], dtype='<U19')




```python
# Loop para substituir o valor ausente (valor_coringa) pelos valores do array de estatísticas
for i in [1,3,4,5]:
    arr_numeric[:,i] = np.where(arr_numeric[:,i] == valor_coringa, 
                                arr_stats[2, colunas_numericas[i]], 
                                arr_numeric[:,i])
```


```python
# Na prática, retirei o valor coringa e coloquei uma estatística
arr_numeric
```




    array([[48010226,    35000,     1000,       28,     1372,    41913],
           [57693261,    35000,     1000,       28,     1372,    41913],
           [59432726,    35000,     1000,       28,     1372,    41913],
           ...,
           [50415990,    35000,     1000,       28,     1372,    41913],
           [46154151,    35000,     1000,       28,     1372,    41913],
           [66055249,    35000,     1000,       28,     1372,    41913]])



# Trabalhando com o segundo dataset


```python
# Carrega o segundo dataset
dados_cot = np.genfromtxt("dataset2.csv", 
                          delimiter = ',', 
                          autostrip = True, 
                          skip_header = 1, 
                          usecols = 3)
```


```python
# Visualiza
dados_cot
```




    array([1.13, 1.12, 1.08, 1.11, 1.1 , 1.12, 1.09, 1.13, 1.13, 1.1 , 1.06, 1.09])




```python
# Nomes de colunas
header_string
```




    array(['issue_date', 'loan_status', 'term_months', 'sub_grade', 'verification_status', 'state_address'], dtype='<U19')




```python
# Dados
arr_strings
```




    array([[ 5,  1, 36, 13,  1,  1],
           [ 0,  1, 36,  5,  1,  4],
           [ 9,  1, 36, 10,  1,  4],
           ...,
           [ 6,  1, 36,  5,  1,  1],
           [ 4,  1, 36, 17,  1,  3],
           [12,  1, 36,  4,  0,  3]])




```python
# A coluna 0 do array de strings é o mês
arr_strings[:,0]
```




    array([ 5,  0,  9, ...,  6,  4, 12])




```python
# Vamos atribuir a coluna de mês à variável chamada exchange_rate
exchange_rate = arr_strings[:,0]
```


```python
exchange_rate
```




    array([ 5,  0,  9, ...,  6,  4, 12])




```python
# Loop para preencher a variável exchange_rate com a taxa correspondente ao mês
# Usamos dados_cot[i - 1] devido a forma como carregamos os meses para comportar o zero
for i in range(1,13):
    exchange_rate = np.where(exchange_rate == i, dados_cot[i - 1], exchange_rate) 
```


```python
exchange_rate
```




    array([1.1 , 0.  , 1.13, ..., 1.12, 1.11, 1.09])




```python
# Onde a taxa de câmbio estiver com zero substituímos pela média
exchange_rate = np.where(exchange_rate == 0, np.mean(dados_cot), exchange_rate)
```


```python
exchange_rate
```




    array([1.1 , 1.11, 1.13, ..., 1.12, 1.11, 1.09])




```python
exchange_rate.shape
```




    (10000,)




```python
arr_numeric.shape
```




    (10000, 6)




```python
exchange_rate = np.reshape(exchange_rate, (10000,1))
```


```python
# Concatenação dos arrays
arr_numeric = np.hstack((arr_numeric, exchange_rate))
```


```python
# Inclui o nome da coluna no array de nomes de colunas
header_numeric = np.concatenate((header_numeric, np.array(['exchange_rate'])))
```


```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate'], dtype='<U19')



## Criando colunas para as taxas de câmbio em USD e EURO.


```python
# Colunas em USD
columns_dollar = np.array([1,2,4,5])
```


```python
# Visualização
arr_numeric[:, 6]
```




    array([1.1 , 1.11, 1.13, ..., 1.12, 1.11, 1.09])




```python
# Shape
arr_numeric.shape
```




    (10000, 7)




```python
# Loop pelas colunas USD para aplicar a taxa de conversão para EURO
for i in columns_dollar:
    arr_numeric = np.hstack((arr_numeric, np.reshape(arr_numeric[:,i] / arr_numeric[:,6], (10000,1))))
```


```python
arr_numeric.shape
```




    (10000, 11)




```python
# Visualização
arr_numeric
```




    array([[48010226.  ,    35000.  ,     1000.  , ...,      912.38,     1251.79,    38240.58],
           [57693261.  ,    35000.  ,     1000.  , ...,      904.42,     1240.86,    37906.76],
           [59432726.  ,    35000.  ,     1000.  , ...,      888.42,     1218.91,    37236.35],
           ...,
           [50415990.  ,    35000.  ,     1000.  , ...,      891.03,     1222.49,    37345.74],
           [46154151.  ,    35000.  ,     1000.  , ...,      899.74,     1234.44,    37710.8 ],
           [66055249.  ,    35000.  ,     1000.  , ...,      914.58,     1254.8 ,    38332.79]])



## Expandindo o cabeçalho com as novas colunas.


```python
header_additional = np.array([column_name + '_EUR' for column_name in header_numeric[columns_dollar]])
```


```python
header_additional
```




    array(['loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'], dtype='<U15')




```python
header_numeric = np.concatenate((header_numeric, header_additional))
```


```python
header_numeric
```




    array(['id', 'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'total_pymnt', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'], dtype='<U19')




```python
header_numeric[columns_dollar] = np.array([column_name + '_USD' for column_name in header_numeric[columns_dollar]])
```


```python
header_numeric
```




    array(['id', 'loan_amnt_USD', 'funded_amnt_USD', 'int_rate', 'installment_USD', 'total_pymnt_USD', 'exchange_rate', 'loan_amnt_EUR', 'funded_amnt_EUR', 'installment_EUR', 'total_pymnt_EUR'],
          dtype='<U19')




```python
columns_index_order = [0,1,7,2,8,3,4,9,5,10,6]
```


```python
header_numeric = header_numeric[columns_index_order]
```


```python
arr_numeric = arr_numeric[:,columns_index_order]
```

# Pré-Processamento da Variável int_rate


```python
header_numeric
```




    array(['id', 'loan_amnt_USD', 'loan_amnt_EUR', 'funded_amnt_USD', 'funded_amnt_EUR', 'int_rate', 'installment_USD', 'installment_EUR', 'total_pymnt_USD', 'total_pymnt_EUR', 'exchange_rate'],
          dtype='<U19')




```python
arr_numeric[:,5]
```




    array([28., 28., 28., ..., 28., 28., 28.])




```python
# Vamos apenas dividir por 100
arr_numeric[:,5] = arr_numeric[:,5] / 100
```


```python
arr_numeric[:,5]
```




    array([0.28, 0.28, 0.28, ..., 0.28, 0.28, 0.28])



# Checkpoint com variáveis numéricas limpas e pré-processadas


```python
checkpoint_numeric = checkpoint("Checkpoint-Numeric", header_numeric, arr_numeric)
```


```python
checkpoint_numeric['header'], checkpoint_numeric['data']
```




    (array(['id', 'loan_amnt_USD', 'loan_amnt_EUR', 'funded_amnt_USD', 'funded_amnt_EUR', 'int_rate', 'installment_USD', 'installment_EUR', 'total_pymnt_USD', 'total_pymnt_EUR', 'exchange_rate'],
           dtype='<U19'),
     array([[48010226.  ,    35000.  ,    31933.3 , ...,    41913.  ,    38240.58,        1.1 ],
            [57693261.  ,    35000.  ,    31654.54, ...,    41913.  ,    37906.76,        1.11],
            [59432726.  ,    35000.  ,    31094.7 , ...,    41913.  ,    37236.35,        1.13],
            ...,
            [50415990.  ,    35000.  ,    31186.05, ...,    41913.  ,    37345.74,        1.12],
            [46154151.  ,    35000.  ,    31490.9 , ...,    41913.  ,    37710.8 ,        1.11],
            [66055249.  ,    35000.  ,    32010.3 , ...,    41913.  ,    38332.79,        1.09]]))



# Construindo o Dataset Final


```python
checkpoint_strings['data'].shape
```




    (10000, 6)




```python
checkpoint_numeric['data'].shape
```




    (10000, 11)




```python
# Concatena os arrays
df_final = np.hstack((checkpoint_numeric['data'], checkpoint_strings['data']))
```


```python
df_final
```




    array([[48010226.  ,    35000.  ,    31933.3 , ...,       13.  ,        1.  ,        1.  ],
           [57693261.  ,    35000.  ,    31654.54, ...,        5.  ,        1.  ,        4.  ],
           [59432726.  ,    35000.  ,    31094.7 , ...,       10.  ,        1.  ,        4.  ],
           ...,
           [50415990.  ,    35000.  ,    31186.05, ...,        5.  ,        1.  ,        1.  ],
           [46154151.  ,    35000.  ,    31490.9 , ...,       17.  ,        1.  ,        3.  ],
           [66055249.  ,    35000.  ,    32010.3 , ...,        4.  ,        0.  ,        3.  ]])




```python
# Verifica se tem valor ausente
np.isnan(df_final).sum()
```




    0




```python
# Concatena os arrays de nomes de colunas
header_full = np.concatenate((checkpoint_numeric['header'], checkpoint_strings['header']))
```


```python
# Ordenando o dataset
df_final = df_final[np.argsort(df_final[:,0])]
```


```python
df_final
```




    array([[  373332.  ,    35000.  ,    31792.25, ...,       21.  ,        0.  ,        1.  ],
           [  575239.  ,    35000.  ,    31792.25, ...,       25.  ,        1.  ,        2.  ],
           [  707689.  ,    35000.  ,    31235.05, ...,       13.  ,        1.  ,        0.  ],
           ...,
           [68614880.  ,    35000.  ,    32010.3 , ...,        8.  ,        1.  ,        1.  ],
           [68615915.  ,    35000.  ,    32010.3 , ...,       10.  ,        1.  ,        2.  ],
           [68616519.  ,    35000.  ,    32010.3 , ...,        3.  ,        0.  ,        2.  ]])




```python
# Conferindo a ordenação da coluna 0
np.argsort(df_final[:,0])
```




    array([   0,    1,    2, ..., 9997, 9998, 9999])



# Gravando o Dataset Final Limpo e Pré-Processado


```python
# Concatena o array de nomes de colunas com o array de dados
df_final = np.vstack((header_full, df_final))
```


```python
# Salva em disco
np.savetxt("dataset_limpo_preprocessado.csv", 
           df_final, 
           fmt = '%s',
           delimiter = ',')
```

# FIM
