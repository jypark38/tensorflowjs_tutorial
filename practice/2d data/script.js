
// car data 가져오기
// 전처리
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car=>({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
    .filter(car => (car.mpg != null && car.horsepower != null))

    return cleaned
}

async function run() {
    // 데이터 가져오고, 학습시킬 데이터를 시각화
    const data = await getData();
    const values = data.map(d => ({
        x: d.horsepower,
        y: d.mpg
    }))

    tfvis.render.scatterplot(
        {name: 'Horsepower v MPG'},
        {values},
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    )

    // 이 패널을 바이저라고 하며 tfjs-vis에서 제공합니다. 이 패널에 시각화를 편리하게 표시할 수 있습니다.

    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    console.log(model)

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model,data,tensorData)
}

// 모델 아키텍처 정의
function createModel(){
    // Create a sequential model
    // 모델 인스턴스화
    // tf.Model 객체를 인스턴스화 한다.
    const model = tf.sequential();

    // Add a single input layer
    // 레이어 추가
    // dense : feed forward net
    // 입력에 숫자가 하나 있으니까 [1] 로 지정
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}))
    model.add(tf.layers.dense({units: 200, activation: 'relu'}));
    // output layer
    model.add(tf.layers.dense({units:1, useBias: true}))

    return model
}

document.addEventListener('DOMContentLoaded',run)

// 데이터를 텐서로 변환하기

// shuffle 과 normalizing을 한다
// y축에 MPG(Miles Per Gallon)을 두고 있습니다.
function convertToTensor(data){
    // 계산을 tidy 안에 래핑하면 중간 텐서를 dispose 한다
    // (tidy 함수는 TensorFlow.js에서 사용되며 자동으로 생성된 중간 텐서를 정리하여 메모리 누수를 방지합니다.)
    // tidy 함수를 사용하면 중간 텐서가 자동으로 해제되므로 메모리를 효과적으로 관리할 수 있습니다.

    return tf.tidy(()=>{
        // 데이터 셔픍 : 필수
        tf.util.shuffle(data)

        // 텐서로 데이터 변환하기
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg)

        const inputTensor = tf.tensor2d(inputs, [inputs.length,1])
        const labelTensor = tf.tensor2d(labels, [labels.length,1])

        // min-max scaling 으로 normalize
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        // (point - min)/(max-min)
        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
          }
    })
}

// model train

async function trainModel(model, inputs, labels){
    // Prepare the model for training
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    })

    const batchSize = 32
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            {height:200, callbacks:['onEpochEnd']}
        )
    })
}

function testModel(model, inputData, normalizationData){
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData

    // 0 에서 1 사이 값을 생성해서 예측한다
    // 스케일링을 해제하고 진행한다.

    const [xs, preds] = tf.tidy(()=>{

        const xs = tf.linspace(0,1,100)
        const preds = model.predict(xs.reshape([100,1]))

        const unNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin)
        const unNormPreds = preds.mul(labelMax.sub(labelMin)).add(labelMin)

        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    })

    const predictedPoints = Array.from(xs).map((val,i)=>{
        return {x: val, y: preds[i]}
    })

    const originalPoints = inputData.map(d => ({
        x: d.horsepower, y: d.mpg
    }))

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'},
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
      height: 300
    }
    )
}