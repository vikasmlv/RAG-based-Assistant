## import dependencies
from datasets import Dataset
from ragas import evaluate
from modules.language_model import ragas_eval_llm, ragas_eval_embedF
from ragas.metrics import faithfulness, answer_correctness
from ragas.run_config import RunConfig
from ast import literal_eval

## load as Dataset -> convert to pandas DataFrame
dataset = Dataset.from_csv("RAGAS-dataset/eval-dataset-final.csv")
df = dataset.to_pandas()

## apply literal eval to unwrap the lists (since it gets stored wrapped around a string while calling .to_csv())
df['retrieved_contexts'] = df['retrieved_contexts'].apply(literal_eval)

## convert back to Dataset for RAGAS compatibility
dataset = Dataset.from_pandas(df)

## evaluation
score = evaluate(
    dataset=dataset,
    llm=ragas_eval_llm,
    embeddings=ragas_eval_embedF,
    metrics=[faithfulness, answer_correctness],
    run_config=RunConfig(timeout=1200)
)

## convert to pandas DataFrame
df = score.to_pandas()

## save it locally
df.to_csv('./RAGAS-dataset/eval-score.csv', index=False)