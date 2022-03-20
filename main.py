from helper import *
from data import DataModule
from fairr_ruleselector_model import FaiRRRuleSelector
from fairr_factselector_model import FaiRRFactSelector
from fairr_reasoner_model import FaiRRReasoner
from proof_inference import FaiRRInference

model_dict = {
	'fairr_ruleselector'      : FaiRRRuleSelector,
	'fairr_factselector'      : FaiRRFactSelector,
	'fairr_reasoner'          : FaiRRReasoner,
	'fairr_inference'         : FaiRRInference,
}

monitor_dict = {
	'fairr_ruleselector'      : ('valid_macro_f1_epoch', 'max'),
	'fairr_factselector'      : ('valid_macro_f1_epoch', 'max'),
	'fairr_reasoner'          : ('valid_acc_epoch', 'max'),
	'fairr_inference'         : ('valid_acc_epoch', 'max'),
}

def generate_hydra_overrides():
	parser = ArgumentParser()
	parser.add_argument('--override')	# Overrides the default hydra config. Setting order is not fixed. E.g., --override rtx_8000,fixed
	args, _ = parser.parse_known_args()

	overrides = []
	if args.override is not None:
		groups = [x for x in os.listdir('./configs/') if os.path.isdir('./configs/' + x)]
		# print(groups)
		for grp in groups:
			confs = [x.replace('.yaml', '') for x in os.listdir('./configs/' + grp) if os.path.isfile('./configs/' + grp + '/' + x)]
			for val in args.override.split(','):
				if val in confs:
					overrides.append(f'{grp}={val}')

	return parser, overrides

def load_hydra_cfg(overrides):
	initialize(config_path="./configs/")
	cfg = compose("config", overrides=overrides)
	print('Composed hydra config:\n\n', OmegaConf.to_yaml(cfg))

	return cfg

def parse_args(args=None):
	override_parser, overrides = generate_hydra_overrides()
	hydra_cfg                  = load_hydra_cfg(overrides)
	defaults                   = dict()
	for k,v in hydra_cfg.items():
		if type(v) == DictConfig:
			defaults.update(v)
		else:
			defaults.update({k: v})

	parser = argparse.ArgumentParser(parents=[override_parser], add_help=False)
	parser = pl.Trainer.add_argparse_args(parser)
	parser = model_dict[defaults['model']].add_model_specific_args(parser)
	parser = DataModule.add_data_specific_args(parser)

	parser.add_argument('--seed', 				default=42, 					type=int,)
	parser.add_argument('--name', 				default='test', 				type=str,)
	parser.add_argument('--log_db', 			default='manual_runs', 			type=str,)
	parser.add_argument('--tag_attrs', 			default='model,dataset,arch', 	type=str,)
	parser.add_argument('--ckpt_path', 			default='', 					type=str,)
	parser.add_argument('--eval_splits', 		default='', 					type=str,)
	parser.add_argument('--debug', 				action='store_true')
	parser.add_argument('--save_checkpoint', 	action='store_true')
	parser.add_argument('--resume_training', 	action='store_true')
	parser.add_argument('--evaluate_ckpt', 		action='store_true')

	parser.set_defaults(**defaults)

	return parser.parse_args()

def get_callbacks(args):

	monitor, mode = monitor_dict[args.model]

	checkpoint_callback = ModelCheckpoint(
		monitor=monitor,
		dirpath=os.path.join(args.root_dir, 'checkpoints'),
		save_top_k=1,
		mode=mode,
		verbose=True,
		save_last=False,
	)

	early_stop_callback = EarlyStopping(
		monitor=monitor,
		min_delta=0.00,
		patience=5,
		verbose=False,
		mode=mode
	)

	return [checkpoint_callback, early_stop_callback]


def main(args, splits='all'):
	pl.seed_everything(args.seed)

	if args.debug:
		# for DEBUG purposes only
		args.limit_train_batches = 10
		args.limit_val_batches   = 10
		args.limit_test_batches  = 10
		args.max_epochs          = 2
		# for DEBUG purposes only

	args.root_dir = f'../saved/{args.name}'
	if args.model == 'fairr_inference':
		os.mkdir(args.root_dir)

	print('Building trainer...')
	trainer	= pl.Trainer.from_argparse_args(
		args,
		callbacks=get_callbacks(args),
		num_sanity_val_steps=0,
	)

	print(f'Loading {args.dataset} dataset')
	dm = DataModule(
			args.dataset,
			args.train_dataset,
			args.dev_dataset,
			args.test_dataset,
			args.arch,
			train_batch_size=args.train_batch_size,
			eval_batch_size=args.eval_batch_size,
			num_workers=args.num_workers,
			pad_idx=args.padding,
		)
	dm.setup(splits=splits)

	print(f'Loading {args.model} - {args.arch} model...')
	if args.model == 'fairr_ruleselector':
		model = model_dict[args.model](
				arch=args.arch,
				train_batch_size=args.train_batch_size,
				eval_batch_size=args.eval_batch_size,
				accumulate_grad_batches=args.accumulate_grad_batches,
				learning_rate=args.learning_rate,
				max_epochs=args.max_epochs,
				optimizer=args.optimizer,
				adam_epsilon=args.adam_epsilon,
				weight_decay=args.weight_decay,
				lr_scheduler=args.lr_scheduler,
				warmup_updates=args.warmup_updates,
				freeze_epochs=args.freeze_epochs,
				gpus=args.gpus,
				hf_name=args.hf_name,
				cls_dropout=args.cls_dropout,
			)

	elif args.model == 'fairr_factselector':
		model = model_dict[args.model](
				arch=args.arch,
				train_batch_size=args.train_batch_size,
				eval_batch_size=args.eval_batch_size,
				accumulate_grad_batches=args.accumulate_grad_batches,
				learning_rate=args.learning_rate,
				max_epochs=args.max_epochs,
				optimizer=args.optimizer,
				adam_epsilon=args.adam_epsilon,
				weight_decay=args.weight_decay,
				lr_scheduler=args.lr_scheduler,
				warmup_updates=args.warmup_updates,
				freeze_epochs=args.freeze_epochs,
				gpus=args.gpus,
				hf_name=args.hf_name,
			)

	elif args.model == 'fairr_reasoner':
		model = model_dict[args.model](
				arch=args.arch,
				train_batch_size=args.train_batch_size,
				eval_batch_size=args.eval_batch_size,
				accumulate_grad_batches=args.accumulate_grad_batches,
				learning_rate=args.learning_rate,
				max_epochs=args.max_epochs,
				optimizer=args.optimizer,
				adam_epsilon=args.adam_epsilon,
				weight_decay=args.weight_decay,
				lr_scheduler=args.lr_scheduler,
				warmup_updates=args.warmup_updates,
				freeze_epochs=args.freeze_epochs,
				gpus=args.gpus,
				hf_name=args.hf_name,
			)

	elif args.model == 'fairr_inference':
		model = model_dict[args.model](
				ruleselector_ckpt=args.ruleselector_ckpt,
				factselector_ckpt=args.factselector_ckpt,
				reasoner_ckpt=args.reasoner_ckpt,
				arch=args.arch,
				train_batch_size=args.train_batch_size,
				eval_batch_size=args.eval_batch_size,
				accumulate_grad_batches=args.accumulate_grad_batches,
				learning_rate=args.learning_rate,
				max_epochs=args.max_epochs,
				optimizer=args.optimizer,
				adam_epsilon=args.adam_epsilon,
				weight_decay=args.weight_decay,
				lr_scheduler=args.lr_scheduler,
				warmup_updates=args.warmup_updates,
				freeze_epochs=args.freeze_epochs,
				gpus=args.gpus,
			)

	return dm, model, trainer


if __name__ == '__main__':
	start_time         = time.time()
	args               = parse_args()
	args.name          = f'{args.model}_{args.dataset}_{args.arch}_{time.strftime("%d_%m_%Y")}_{str(uuid.uuid4())[: 8]}'

	# sanity check
	if args.resume_training:
		assert args.ckpt_path != ''
	if args.evaluate_ckpt:
		if args.model == 'fairr_inference':
			pass
		else:
			assert args.ckpt_path != ''
		assert args.eval_splits != ''

	# Update trainer specific args that are used internally by Trainer (which is initialized from_argparse_args)
	args.precision = 16 if args.fp16 else 32
	if args.resume_training:
		args.resume_from_checkpoint = args.ckpt_path

	# Load the datamodule, model, and trainer used for training (or evaluation)
	if not args.evaluate_ckpt:
		dm, model, trainer = main(args)
	else:
		dm, model, trainer = main(args, splits=args.eval_splits.split(','))

	print(vars(args))

	if not args.evaluate_ckpt:
		# train the model from scratch (or resume training from the checkpoint)
		trainer.fit(model, dm)
		print('Testing the best model...')
		trainer.test(ckpt_path='best', dataloaders=trainer.datamodule.test_dataloader())
		if not args.save_checkpoint:
			os.remove(trainer.checkpoint_callback.best_model_path)
	else:
		# evaluate the pretrained model on the provided splits
		if args.model == 'fairr_inference':
			model_ckpt = model
		else:
			model_ckpt = model.load_from_checkpoint(args.ckpt_path)
		print('Testing the best model...')
		for split in args.eval_splits.split(','):
			print(f'Evaluating on split: {split}')
			if split == 'train':
				loader = dm.train_dataloader(shuffle = False)
			elif split == 'dev':
				loader = dm.val_dataloader()
			elif split == 'test':
				loader = dm.test_dataloader()

			trainer.test(model=model_ckpt, dataloaders=loader)

	print(f'Time Taken for experiment {args.name}: {(time.time()-start_time) / 3600}h')
