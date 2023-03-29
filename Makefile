run_preprocess:
	python -c 'from FF2S_prod.interface.main import preprocess; preprocess()'

run_clear:
	python -c 'from FF2S_prod.interface.main import empty; empty()'

run_split:
	python -c 'from FF2S_prod.interface.main import train_test_split; train_test_split()'

run_train:
	python -c 'from FF2S_prod.interface.main import train; train()'

run_pred:
	python -c 'from FF2S_prod.interface.main import pred; pred()'

run_all: run_preprocess run_clear run_split run_train run_pred

run_trade_pred: run_train run_pred
