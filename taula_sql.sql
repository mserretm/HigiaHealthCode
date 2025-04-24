CREATE TABLE public.f_cmbd_ha_deeplearning (
	cas varchar(25) NOT NULL,
	dx_revisat text NULL,
	edat int4 NULL,
	genere varchar(1) NULL,
	c_alta varchar(1) NULL,
	periode varchar(4) NULL,
	servei varchar(10) NULL,
	motiuingres text NULL,
	malaltiaactual text NULL,
	exploracio text NULL,
	provescomplementariesing text NULL,
	provescomplementaries text NULL,
	evolucio text NULL,
	antecedents text NULL,
	cursclinic text NULL,
	us_estatentrenament int4 DEFAULT 0 NULL,
	us_registre varchar(1) NULL,
	dx_prediccio text NULL,
	us_dataentrenament timestamp NULL,
	CONSTRAINT f_cmbd_ha_deeplearning_pk PRIMARY KEY (cas)
);



CREATE TABLE public.f_deeplearning_evaluate (
	id serial4 NOT NULL,
	experimento_id varchar(100) NOT NULL,
	caso_id varchar(100) NOT NULL,
	codis_reals _text NULL,
	codis_predits_top15 _text NULL,
	probs_predits_top15 _float4 NULL,
	codis_predits_confianza_top15 _text NULL,
	ordre_real _text NULL,
	ordre_predit _text NULL,
	accuracy float4 NULL,
	precisi float4 NULL,
	recall float4 NULL,
	f1_score float4 NULL,
	order_accuracy float4 NULL,
	kendall_tau float4 NULL,
	code_loss float4 NULL,
	order_loss float4 NULL,
	num_codis_reals int4 NULL,
	num_codis_predits_top15 int4 NULL,
	num_codis_predits_confianza_top15 int4 NULL,
	ver_modelo varchar(50) NULL,
	set_validacion varchar(100) NULL,
	ts_inici timestamp NULL,
	ts_final timestamp NULL,
	CONSTRAINT f_deeplearning_evaluate_pkey PRIMARY KEY (id)
);



CREATE TABLE public.f_deeplearning_train (
	id serial4 NOT NULL,
	cas varchar(25) NOT NULL,
	data_entrenament date NOT NULL,
	hora_inici varchar(8) NOT NULL,
	hora_fi varchar(8) NOT NULL,
	durada varchar(8) NOT NULL,
	lr_inicial float8 NOT NULL,
	decay float8 NOT NULL,
	batch_size int4 NOT NULL,
	max_epochs int4 NOT NULL,
	early_stopping_patience int4 NOT NULL,
	max_len_input int4 NOT NULL,
	umbral_confianza_prediccio float8 NOT NULL,
	epochs int4 DEFAULT 0 NOT NULL,
	lr_final float8 DEFAULT 0 NOT NULL,
	loss_total_final float8 DEFAULT 0 NOT NULL,
	loss_code_final float8 DEFAULT 0 NOT NULL,
	loss_order_final float8 DEFAULT 0 NOT NULL,
	CONSTRAINT f_deeplearning_train_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_f_deeplearning_train_cas ON public.f_deeplearning_train USING btree (cas);
CREATE INDEX idx_f_deeplearning_train_data ON public.f_deeplearning_train USING btree (data_entrenament);