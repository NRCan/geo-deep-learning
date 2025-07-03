
## **GDL Refactoring: What was that all about and next steps?**

**Refactoring GDL with PyTorch Lightning**

![lightning Overview](https://github.com/valhassan/geo-deep-learning/blob/573-feature-refactor-geo-deep-learning/images/lightning_overview.png?raw=true)


Since 2020:

![lightning Issues](https://github.com/valhassan/geo-deep-learning/blob/422808634c0446a10efd4bb93ccb506c62deeb30/images/lightning_issues_on_github.png?raw=true)

**Why did we decide to refactor?**

- Access to bleeding edge features.

![lightning Features](https://github.com/valhassan/geo-deep-learning/blob/422808634c0446a10efd4bb93ccb506c62deeb30/images/lightning_features.png?raw=true)
- Less Brittle Code.
- Maintainability.
- Upgrade to industry standards. TorchGeo is built from the ground up with Lightning.

**4 years has seen consolidation and maturity in the Pytorch Ecosystem.**
<div align="left">
      <a href="https://youtu.be/rgP_LBtaUEc?si=4F7P9YbT70EhCqVP">
         <img src="https://i.ytimg.com/vi/rgP_LBtaUEc/maxresdefault.jpg" style="width:100%;">
      </a>
</div>

## **Building Blocks**
![lightning Architecture](https://github.com/valhassan/geo-deep-learning/blob/573-feature-refactor-geo-deep-learning/images/GDL_lightning_architecture.png?raw=true)


## Next Steps

Collective effort is required!

- Refactor branch created; the new code base will live there until we are ready to merge squatch "develop branch"!
- Project will be created on Github to manage milestones and tasks.
- Documentation, Lightning modules as applicable, Some tests, CI/CD Github workflow, e.t.c are required.
