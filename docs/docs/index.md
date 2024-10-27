# Sex Assigned at Birth documentation!

## Description

El proyecto busca desarrollar un modelo para predecir el sexo al nacer de los usuarios de un neobanco en Colombia, basado en los nombres registrados en la aplicación.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://co-miad4304-data-687973248987-dev/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://co-miad4304-data-687973248987-dev/data/` to `data/`.


