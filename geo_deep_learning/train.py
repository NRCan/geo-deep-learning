from lightning.pytorch.cli import ArgsType, LightningCLI

def main(args: ArgsType = None) -> None:
    LightningCLI(
        save_config_kwargs={"overwrite": True},
        args=args,
    )

if __name__ == "__main__":
    main()
    print("Done!")