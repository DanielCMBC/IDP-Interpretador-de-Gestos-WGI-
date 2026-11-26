# IDP - Interpretador de Gestos Web (WGI)
1.2026 - Trabalho da Disciplina de Oficina em Soluções Web

WGI: Web Gesture Interpreter

O Web Gesture Interpreter (WGI) é uma extensão inovadora projetada para transformar a maneira como interagimos com o navegador. Ao converter movimentos manuais capturados via webcam em comandos de navegação, o projeto elimina a barreira do contato físico com periféricos, oferecendo uma experiência de usuário fluida, intuitiva e futurista.

Visão Geral do Projeto
A proposta central do WGI é democratizar o controle por gestos no ecossistema web. Seja para facilitar o consumo de conteúdo em momentos de multitarefa  como seguir uma receita na cozinha ou realizar uma apresentação  ou para servir como uma ferramenta crítica de acessibilidade, o interpretador traduz a linguagem corporal em ações digitais em tempo real.

Proposta de Valor e Diferenciais
O WGI não é apenas uma alternativa ao mouse, mas uma nova camada de interação que se destaca por:

Acessibilidade Inclusiva: Oferece autonomia para usuários com limitações motoras que possuem dificuldade no manuseio de dispositivos convencionais.

Navegação Sem Contato (Touchless): Ideal para ambientes onde a higiene ou a distância do dispositivo são prioridades.

Interatividade em Mídias Sociais: Permite o consumo de feeds verticais e mídias através de "swipes" e comandos gestuais naturais.

Expansão para Imersão: Serve como uma ponte de baixo custo para experiências de Realidade Aumentada (AR) diretamente no browser, sem a necessidade de hardware caro.

Desde o controle de câmeras em conferências remotas até a navegação em jogos baseados em web, o potencial do WGI se estende a qualquer aplicação que se beneficie de uma interface espacial. Ele transforma o navegador de uma janela estática em um ambiente responsivo ao movimento humano.

## Como usar

1. Abra o servidor WGI (o app compilado em `wgi-servidor/`, ou `python back-end/wgi.py` a partir do código-fonte).
2. Abra a extensão, escolha o modo de controle (**Mãos**, **Olhos** ou **Tronco**) e clique em **Start**.
3. O modelo do MediaPipe só é carregado depois que você escolhe o modo: o servidor responde na hora e carrega o modelo em segundo plano enquanto abre a câmera. A extensão mostra o progresso ("Loading model…", centralização, "Tracking").

Trocar de modo durante o rastreamento troca o modelo ao vivo: o anterior é fechado e a câmera é liberada antes de o novo começar.

## Controles

### Mãos (MediaPipe Hands)

A palma aberta move o cursor. Os cliques são feitos encostando a ponta do dedo no polegar (pode ser na polpa do dedo, não precisa ser ponta com ponta).

| Gesto | Ação |
|---|---|
| Polegar + indicador, toque | Clique esquerdo |
| Polegar + indicador, dois toques rápidos | Clique duplo |
| Polegar + indicador, encostar e **mover** a mão | Arrastar (abrir os dedos solta) |
| Polegar + médio, toque | Clique direito |
| Polegar + médio, encostar e mover para cima/baixo | Rolar a página |

Sem calibração: a distância de cada dedo até o polegar é medida em relação ao tamanho da própria mão e comparada com a distância "aberta" daquele dedo, aprendida continuamente. Um toque é um toque sem mover a mão, não importa a duração. O cursor é suavizado por um filtro One Euro num laço próprio de ~120 Hz, independente da câmera (~30 fps).

### Olhos (MediaPipe Face Mesh + íris)

A direção da cabeça move o cursor (rotação estimada a partir de ~33 pontos rígidos do rosto: testa, contorno superior e dorso do nariz). Olhe para o centro da tela durante a centralização (~1 s); **Re-center** refaz a centralização.

| Gesto | Ação |
|---|---|
| Piscar só o olho **esquerdo** | Clique esquerdo |
| Piscar o olho esquerdo duas vezes | Clique duplo |
| Manter o olho esquerdo fechado (~0,6 s) e virar a cabeça | Arrastar |
| Piscar só o olho **direito** | Clique direito |
| Manter a boca aberta e acenar com a cabeça | Rolar a página |
| Piscar normalmente (os dois olhos) | Ignorado |
| Rir, tossir, espirrar | Cursor e cliques pausados por um instante |

### Tronco (MediaPipe Pose)

| Gesto | Ação |
|---|---|
| Inclinar para os lados / endireitar-se | Mover o cursor |
| Ficar parado ~1,2 s | Clique esquerdo |
| Encolher os ombros | Clique direito |
| Inclinar para a frente / para trás | Rolar a página |

Os ajustes de olhos e tronco ficam em `calibration.json` (seções `"eyes"` e `"torso"`) e podem ser alterados sem recompilar.

## Avaliação: precisão, revocação, F1 e acurácia

Precisão e revocação exigem saber o que o usuário **quis** fazer, e as gravações reais só registram o que o sistema **fez**. Por isso a avaliação usa um benchmark sintético rotulado (`evaluation/`): centenas de episódios com rótulo conhecido (clique, clique duplo, clique direito, arrastar, rolar e "nada", com casos que não devem disparar: mão relaxada, punho, quase-toque, movimento rápido, piscadas, virar a cabeça, olhar para baixo, risada, fala). Os parâmetros foram **medidos em sessões reais** gravadas com `WGI_TRACE`:

- ruído dos pontos da mão: 0,95 mm por eixo por quadro
- acoplamento dos dedos: o anelar vai junto com o médio, e o médio com o indicador
- deriva da mão durante um toque
- profundidade real das flexões de dedo
- tremor da abertura dos olhos
- o outro olho semicerrando enquanto um pisca

Cada estágio reproduz um experimento do projeto. Métricas: média macro sobre as 5 classes de gesto; "disparo falso" é a fração de episódios "nada" que dispararam alguma ação (60 episódios por classe, semente 7).

### Mãos

| Estágio | O que mudou | Acurácia | Precisão | Revocação | F1 | Disparo falso |
|---|---|---|---|---|---|---|
| V0 | Original: ângulos de flexão dos dedos | 39,7% | 86,2% | 34,3% | 43,1% | 33,3% |
| S1 | Pinça ponta-com-ponta, soltar fixo, segurar = arrastar, filtro de punho | 60,3% | 84,6% | 52,3% | 63,2% | 0,0% |
| S2 | + contato pela polpa do dedo | 81,9% | 92,1% | 78,3% | 80,5% | 0,0% |
| S3 | + arrastar = mover a mão (não segurar) | 92,8% | 100,0% | 91,3% | 95,0% | 0,0% |
| S4 | + soltar relativo ao ponto mais fundo, re-armar | 92,8% | 100,0% | 91,3% | 95,0% | 0,0% |
| S5 | + sem filtro de punho, sem arrastar/rolar no "tranco" ao soltar (**atual**) | 100,0% | 100,0% | 100,0% | 100,0% | 0,0% |

### Olhos

| Estágio | O que mudou | Acurácia | Precisão | Revocação | F1 | Disparo falso |
|---|---|---|---|---|---|---|
| E1 | Razão de aspecto clássica do olho, limiares rasos, "outro olho fechado" = piscada | 31,4% | 20,0% | 20,0% | 20,0% | 11,7% |
| E2 | + abertura medida pela distância entre os olhos (imune a virar a cabeça) | 34,4% | 40,0% | 21,3% | 22,5% | 0,0% |
| E3 | + limiares mais fundos (0,5 / 0,7), nível de abertura pela mediana | 56,7% | 90,0% | 48,0% | 57,3% | 0,0% |
| E4 | + piscada avaliada em relação ao outro olho (**atual**) | 96,9% | 98,2% | 97,3% | 97,7% | 5,0% |

### Como ler estes números

- **É um limite superior, não uma promessa.** Os cenários sintéticos não têm desfoque de movimento, oclusão nem perdas de rastreamento, e foram montados depois de ajustar a lógica nas mesmas gravações. O ganho *relativo* entre estágios é o resultado confiável; os 100% do S5 não são.
- **V0 falha por "não disparar":** as flexões medidas do usuário não chegam aos limiares padrão (por isso a revocação é baixa), e 1 em cada 3 episódios "nada" ainda disparou algo.
- **S4 = S3 no benchmark** porque não há um cenário de "toques rápidos repetidos sem abrir a mão", que é exatamente o que o S4 corrigiu nas gravações reais.
- **E4 tem um custo real:** reconhece quase todas as piscadas de um olho só, mas uma piscada normal em que um olho fecha um quadro antes do outro às vezes vira clique esquerdo (3 de 60 episódios "nada").
- A segunda semente (11) dá resultados equivalentes (mãos S5 99,4%, olhos E4 97,5%).

Para reproduzir:

```
python evaluation/benchmark_hands.py --episodes 60 --seed 7
python evaluation/benchmark_eyes.py  --episodes 60 --seed 7
```

## Ferramentas de desenvolvimento

- `WGI_TRACE=caminho.csv` grava por quadro só números (ângulos da cabeça, abertura dos olhos, distâncias dedo-polegar, posições dos pontos da mão) para ajustar o comportamento reproduzindo sessões reais. Nenhuma imagem é gravada.
- `python back-end/wgi.py --start --mode eyes --show` inicia direto, sem a extensão, com a janela de depuração.
- Endpoints: `GET /state`, `GET /config`, `POST /start_tracking {"mode": "hands|eyes|torso"}`, `POST /stop_tracking`, `POST /recenter`, `POST /config`.

## macOS / Apple Silicon

Para compilar e executar o backend em Macs M1/M2/M3:

- Use Python arm64 nativo. Confirme com `python3 -c "import platform; print(platform.machine())"`; o resultado deve ser `arm64`.
- Use Python 3.10 ou 3.11 para manter compatibilidade com MediaPipe.
- De permissao de Camera e Accessibility ao Terminal ou ao app compilado em System Settings > Privacy & Security.
- O backend tenta abrir a webcam com AVFoundation no macOS e volta para o backend padrao do OpenCV se necessario.
- O arquivo `calibration.json` e salvo ao lado do executavel quando o backend esta compilado.

## Linux / X11 / Wayland

O backend tenta abrir a webcam no Linux usando V4L2 e volta para o backend padrao do OpenCV se necessario.

- Em sessoes X11/Xorg, o controle do cursor usa PyAutoGUI. Se o PyAutoGUI falhar, o backend tenta usar `xdotool` como fallback para movimento, clique esquerdo, clique direito, clique duplo, drag e scroll.
- Instale `xdotool` no Linux X11 quando quiser o fallback: `sudo apt install xdotool`.
- Em Wayland, a camera pode funcionar, mas muitos compositores bloqueiam controle global do cursor por seguranca. Para suporte completo de clique/movimento, use uma sessao X11/Xorg ou configure uma ferramenta de injecao permitida pelo compositor.
- O endpoint `/state` inclui `display_server` no Linux para ajudar a diagnosticar se a sessao atual e `x11`, `wayland` ou `unknown`.
