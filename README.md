# Machine Learning Applications on Dota 2

The suggestion model is currently being migrated

Models:
* train_hero_embeddings.py : model for representing Dota 2 heroes in a latent space, includes logger and saver utilities.
* train_predictive_model.py : model for predicting Dota 2 match outcomes using the pre-trained hero embeddings. Includes utilities.

Other Utilities:
* OpenDotaAPI.py : Partial Python wrapper for OpenDota API

Data:
* pro_dump.p : professional match data until Aug. 17. In the format: [match_id, start_time, radiant_win, [heroes]]
* final_embeddings : trained embeddings on pro_dump.p


