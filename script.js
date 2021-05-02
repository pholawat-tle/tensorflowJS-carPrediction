const loadingDiv = document.querySelector('.loading');
const mainApp = document.querySelector('main');

async function getData() {
    const carsDataResponse = await fetch(
        'https://storage.googleapis.com/tfjs-tutorials/carsData.json'
    );
    const carsData = await carsDataResponse.json();
    const cleaned = carsData
        .map((car) => ({
            mpg: car.Miles_per_Gallon,
            horsepower: car.Horsepower,
        }))
        .filter((car) => car.mpg != null && car.horsepower != null);

    return cleaned;
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    model.add(tf.layers.dense({ units: 50, activation: 'sigmoid' }));
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}

function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data);

        const inputs = data.map((d) => d.horsepower);
        const labels = data.map((d) => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor
            .sub(labelMin)
            .div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        };
    });
}

async function trainModel(model, inputs, labels) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
    });
}

async function run() {
    const data = await getData();
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    const model = createModel();

    await trainModel(model, inputs, labels);

    // DOM
    loadingDiv.style.display = 'none';
    mainApp.style.display = 'flex';

    document
        .querySelector('#prediction-input')
        .addEventListener('submit', (e) => {
            e.preventDefault();
            const { inputMax, inputMin, labelMax, labelMin } = tensorData;
            const [pred] = tf.tidy(() => {
                const xs = tf
                    .tensor([parseInt(e.target.elements['horsepower'].value)])
                    .sub(inputMin)
                    .div(inputMax.sub(inputMin));

                const pred = model.predict(xs);
                const unNormPred = pred
                    .mul(labelMax.sub(labelMin))
                    .add(labelMin);

                // Un-normalize the data
                return [unNormPred.dataSync()];
            });

            alert(`Predicted Miles Per Gallon = ${pred[0]}`);
        });
}

document.addEventListener('DOMContentLoaded', run);
