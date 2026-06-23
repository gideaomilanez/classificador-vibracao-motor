# Dados de Vibração

Dois arquivos CSV com sinais de vibração coletados de um motor elétrico em duas condições operacionais, utilizados para treinamento e avaliação do classificador Random Forest.

## Arquivos

| Arquivo | Condição | Amostras | Duração |
|---|---|---|---|
| `motor_balanceado.csv` | Motor em condição normal, sem adição de massa | 99.999 | ~1.700 s |
| `motor_desbalanceado.csv` | Motor com parafuso adicionado ao rotor para induzir desbalanceamento | 99.999 | ~1.700 s |

## Formato

Cada arquivo contém um cabeçalho e uma linha por amostra, no formato:

```
timestamp,x1,y1,z1,x2,y2,z2,x3,y3,z3
```

| Coluna | Tipo | Descrição |
|---|---|---|
| `timestamp` | float (Unix) | Instante de coleta em segundos (epoch Unix) |
| `x1`, `y1`, `z1` | float (g) | Aceleração nos eixos X, Y, Z — Sensor 1 (MPU-9250, endereço 0x68) |
| `x2`, `y2`, `z2` | float (g) | Aceleração nos eixos X, Y, Z — Sensor 2 (MPU-9250, endereço 0x69) |
| `x3`, `y3`, `z3` | float (g) | Aceleração nos eixos X, Y, Z — Sensor 3 (MPU-6050, endereço 0x68) |

Valores em unidade de aceleração gravitacional **g** (fundo de escala ±16 g, resolução ≈ 0,061 mg/LSB na configuração ±2 g do MPU-6050).

## Parâmetros de aquisição

- **Frequência de amostragem efetiva:** ≈ 59 Hz
- **Protocolo de transmissão:** Serial UART 115200 baud, formato CSV, 9 campos por linha
- **Hardware:** ESP32 + 2× MPU-9250 + 1× MPU-6050

## Janelamento para treinamento

Os scripts de treinamento utilizam:

- **Tamanho da janela:** 256 amostras (≈ 4,3 s)
- **Passo:** 256 amostras (sem sobreposição)
- **Janelas por arquivo:** 390
- **Total de amostras rotuladas:** 780 (390 por classe)

## Como coletar novos dados

1. Grave o firmware em [`../firmware/firmware_esp32.ino`](../firmware/firmware_esp32.ino) no ESP32
2. Conecte via USB e abra a aplicação [`../scripts/04_monitor_vibracao.py`](../scripts/04_monitor_vibracao.py)
3. Na aba lateral, defina o número de amostras desejado e clique em **Iniciar Coleta**
4. Ao terminar, clique em **Salvar CSV**
5. Renomeie o arquivo conforme a condição coletada e coloque-o nesta pasta
6. Atualize o dicionário `ARQUIVOS` em `02_treinar_modelo.py` com o novo nome
