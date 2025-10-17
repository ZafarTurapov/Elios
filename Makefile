.PHONY: snapshot deploy run dry
snapshot:
	ops/scripts/capture_env.sh
deploy:
	bin/deploy.sh
run:
	PYTHONPATH=/root/stockbot bin/execute_trades.sh
dry:
	ELIOS_DRY_RUN=1 PYTHONPATH=/root/stockbot bin/execute_trades.sh
