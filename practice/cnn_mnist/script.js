import { MnistData } from "./data.js";

/*  
    Mnist 데이터 클래스
    nextTrainBatch(batchSize) : train set에서 이미지와 레이블의 무작위 batch 반환
    nextTestBatch(batchSize) : test set
*/

async function showExamples(data){
    // visor 안에 컨테이너 생성
    const surface = tfvis.visor().surface({
        name: 'Input Data Example',
        tab: 'Input Data'
    })

    // example 얻기
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];

    // 각 예제를 렌더링할 캔버스 요소 생성
    for(let i = 0; i < numExamples; i++){
        const imageTensor = tf.tidy(()=>{
            // 28x28로 리쉐입
            return examples.xs.slice([i,0],[1, examples.xs.shape[1]]).reshape([28,28,1])
        })

        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;'
        await tf.browser.toPixels(imageTensor, canvas);
        surface.drawArea.appendChild(canvas)

        imageTensor.dispose();
    }
}

// model architecture
function getModel() {
    const model = tf.sequential();

    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    /*
    cnn 첫번째 레이어에서, input shape을 명시할거임
    이 레이어에서 convolution 연산에 대한 파라미터를 명시한다.
    */
    /* 
    * @params
    * inputShape : 첫 레이어로 전달될 데이터 모양 -> 흑백 이미지 [28,28,1]
    * 배치크기는 지정하지 않았음
    * -> 배치 크기의 제약이 없도록 설계되어서 추론 중에 모든 배치 크기의 텐서를 전달할 수 없다
    * kernelSize : 입력 데이터에 적용되는 슬라이딩 컨벌루션 필터의 크기
    * filters : 레이어 인풋에 적용할 필터의 수
    * strides : 컨벌루션 연산의 보폭
    * activation : 활성화함수
    * kernelInitializer : 가중치 초기화 메서드
     */
    model.add(tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))

    // max pooling 레이어는 평균 대신 그 영역의 최대값으로 다운샘플링을 한다
    model.add(tf.layers.maxPooling2d({
        poolSize:[2,2],
        strides:[2,2]
    }))

    // 추가적인 conv2d + maxPooling 반복해서 쌓음
    // 필터가 더 많다
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }))
    model.add(tf.layers.maxPooling2d({
        poolSize:[2,2],
        strides:[2,2]
    }))

    // 2d 필터의 아웃풋을 1d 벡터로 펼친다
    // 최종 classifier 레이어로 전달하기 전에 벡터로 펼친다.
    model.add(tf.layers.flatten());

    // Our last layer is a dense layer which has 10 output units, one for each
    // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).

    // 레이블에 대한 확률분포 계산
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'
    }));

    // Choose an optimizer, loss function and accuracy metric,
    // then compile and return the model

    // optimizer : adam
    // loss function : categoricalCrossentropy
    const optimizer = tf.train.adam();
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
}

async function train(model, data){

    const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
    const container = {
        name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
    }
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;


    const [trainXs, trainYs] = tf.tidy(()=>{
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE,28,28,1]),
            d.labels
        ]
    })

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });
    
    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: fitCallbacks
    });
}

// evaluation
const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500){
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const testData = data.nextTestBatch(testDataSize);
    const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH,IMAGE_HEIGHT,1]);
    const labels = testData.labels.argMax(-1);
    const preds = model.predict(testxs).argMax(-1);

    testxs.dispose();
    return [preds, labels]
}

async function showAccuracy(model, data) {
    const [preds, labels] = doPrediction(model, data);
    // acc 계산
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    const container = {name: 'Accuracy', tab: 'Evaluation'};
    tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  
    labels.dispose();
  }
  
  async function showConfusion(model, data) {
    const [preds, labels] = doPrediction(model, data);

    // confusionMatrix
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
    tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});
  
    labels.dispose();
  }

async function run() {
    const data = new MnistData();
    await data.load();
    await showExamples(data);

    const model = getModel();
    tfvis.show.modelSummary({name:'Model Architecture', tab:'Model'}, model);

    await train(model,data);
    await showAccuracy(model, data);
    await showConfusion(model, data);
}

document.addEventListener('DOMContentLoaded', run);