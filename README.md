# Bluesight API Platform

Bluesight API Platform is a platform for geospacial foundational models fine-tuning and inference.

[![Build, push to registry and deploy](https://github.com/bluesightai/clay/actions/workflows/build-and-deploy.yml/badge.svg)](https://github.com/bluesightai/clay/actions/workflows/build-and-deploy.yml)

## Structure

- [app/](app/) is a FastAPI application, which serves as the main API
- [clay/](clay/) is a module for training and inference of Clay foundation model. It will be moved to a separate repository in the future
- [scripts/](scripts/) is a folder with scripts for running calculations which are not part of the API

## Installation

To run the API locally, clone the repository, specify Supabase envs, build the Docker image and run the container:

```bash
echo "SUPABASE_URL=supabase_url" > .env
echo "SUPABASE_KEY=supabase_key" >> .env
docker compose up -d --build
```

## ToDo

- Run inference using files
- Make everything except `pixels` optional on inference API
- Try [BackgroundTasks](https://fastapi.tiangolo.com/tutorial/background-tasks/) instead of `asyncio.create_task` in training job creation
  - Be careful, logging might break it (or it may break itself)
- If update crud schema contains several optional fields and only one is set, the rest are set to null in the update in the database
- Files are loaded fully to RAM on server on upload (is it? debug)
- Move fine-tunings to `clay` module
- Add auth and RLS to database
- Calculate means and stds on train, save and use on inference
- Move `pytorch` to separate poetry group
- Maybe specify PID in logging messages
- Docs
  - Finish `/embeddings` and `/models` docs
  - Return link to an object in the response schema like in [OpenAI API](https://platform.openai.com/docs/api-reference/fine-tuning/create)
  - Return `null` values for key observability ([FastAPI issue](https://github.com/fastapi/fastapi/pull/3770))
