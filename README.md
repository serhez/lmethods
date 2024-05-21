# LMethods
A collection of both novel and baseline methods to enhance the capabilities of language models.

## Developing new methods
If you are developing an existing baseline method, feel free to contribute directly to this repo by creating a new branch with the name of the new method. If you are developing a novel method you would like to keep private for a while, you can fork this repo and make it private; when the new method is ready for publication, you can make a pull request in this repo to merge your method here. You can always `sync` the fork with the upstream (i.e., this repo) in order to get the latest updates either in the GitHub website or via the GitHub CLI as `gh repo sync owner/cli-fork -b BRANCH_NAME`. You can install your own version of the forked package as `pip install git+https://github.com/<USER>/lmethods.git@<BRANCH>` to test your WIP methods in other code bases.

While new methods can be as flexible as needed for your own particular settings and tasks, it is required for them to implement the `Method` interface. Additional functionality may be included in your method's class to comply to your needs. Following this interface guarantees others' ability to test your method as a baseline against their own methods in future research endeavors. If you are unhappy with the current `Method` interface, please open an issue with your proposed changes.

## Integration with other libraries
This package depends on external representations of data sources and language models. `LMethods` is quite flexible with respect to these representations, and we provide always simple interfaces that can be implemented by the user to wrap their existing code and integrate it with this library. Nonetheless, these interfaces are designed after the following libraries (hence, their usage is recommended):

- [`ldata`](https://github.com/serhez/ldata): this library provides dataset and benchmark abstractions that make training, fine-tuning and evaluation of language-models easier.
- [`lmodels`](https://github.com/serhez/lmodels): this package offers access to different LLMs under a common API; you can load existing weights, train, fine-tune or even create your own models within a private fork of this repository.

> [!NOTE]
> While you do not have to use `ldata` objects to use any part of our library, it currently is listed as a dependency of this package because we use their types. In the future, we plan on dropping this dependency to make this package more self-contained and light-weight.

## Todo's
- [ ] Composition of methods (e.g., adding Self-Consistency to any other existing method).
