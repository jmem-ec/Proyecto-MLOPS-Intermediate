from invoke import task

@task
def ingest(ctx):
    ctx.run("ingest")

@task
def clean(ctx):
    ctx.run("python src/data_eng/stage2_cleaning.py")

@task
def features(ctx):
    ctx.run("python src/data_eng/stage3_labeling.py")

@task
def split(ctx):
    ctx.run("python src/data_eng/stage4_splitting.py")

@task
def data_eng_pipeline(ctx):
    ingest(ctx)
    clean(ctx)
    features(ctx)
    split(ctx)

@task
def git(ctx, mensaje):
    ctx.run(f"git add .")
    ctx.run(f"git commit -m '{mensaje}'")
    ctx.run(f"git push origin main")

