# LMethods
A collection of both novel and baseline methods to enhance the capabilities of language models, by Aalto LaReL research group.

## Developing new methods
If you are developing an existing baseline method, feel free to contribute directly to this repo by creating a new branch with the name of the new method. If you are developing a novel method you would like to keep private for a while, you can fork this repo and make it private; when the new method is ready for publication, you can make a pull request in this repo to merge your method here. You can always `sync` the fork with the upstream (i.e., this repo) in order to get the latest updates either in the GitHub website or via the GitHub CLI as `gh repo sync owner/cli-fork -b BRANCH_NAME`.

While new methods can be as flexible as needed for your own particular settings and tasks, it is required for them to implement the `Method` interface. Additional functionality may be included in your method's class to comply to your needs. Following this interface guarantees others' ability to test your method as a baseline against their own methods in future research endeavors. If you are unhappy with the current `Method` interface, please open an issue with your proposed changes.
