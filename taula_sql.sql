-- public.f_cmbd_ha_deeplearning definition

-- Drop table

-- DROP TABLE public.f_cmbd_ha_deeplearning;

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