 # Inline TODOs


# HACK

## [`maze_dataset/benchmark/config_sweep.py`](/maze_dataset/benchmark/config_sweep.py)

- sort by grid size  
  local link: [`/maze_dataset/benchmark/config_sweep.py:282`](/maze_dataset/benchmark/config_sweep.py#L282) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L282](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L282)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=sort%20by%20grid%20size&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L282%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L282%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09sorted%28%0A%09%09%09%09self.result_values.items%28%29%2C%0A%09%09%09%09%23%20HACK%3A%20sort%20by%20grid%20size%0A%09%09%09%09%23%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C--%3C%20name%20of%20config%0A%09%09%09%09%23%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7C%20%20%20%20%7C-----------%3C%20gets%20%27g%7Bn%7D%27%0A%60%60%60&labels=HACK)

  ```python
  sorted(
  	self.result_values.items(),
  	# HACK: sort by grid size
  	#                 |--< name of config
  	#                 |    |-----------< gets 'g{n}'
  ```


- big hassle to do this without a lambda, is it really that bad?  
  local link: [`/maze_dataset/benchmark/config_sweep.py:509`](/maze_dataset/benchmark/config_sweep.py#L509) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L509](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L509)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=big%20hassle%20to%20do%20this%20without%20a%20lambda%2C%20is%20it%20really%20that%20bad%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L509%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L509%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09results_filtered%3A%20SweepResult%20%3D%20results_epkw.get_where%28%0A%09%09%09%09%22maze_ctor%22%2C%0A%09%09%09%09%23%20HACK%3A%20big%20hassle%20to%20do%20this%20without%20a%20lambda%2C%20is%20it%20really%20that%20bad%3F%0A%09%09%09%09lambda%20x%3A%20x.__name__%20%3D%3D%20gen_func%2C%20%20%23%20noqa%3A%20B023%0A%09%09%09%29%0A%60%60%60&labels=HACK)

  ```python
  results_filtered: SweepResult = results_epkw.get_where(
  	"maze_ctor",
  	# HACK: big hassle to do this without a lambda, is it really that bad?
  	lambda x: x.__name__ == gen_func,  # noqa: B023
  )
  ```




## [`maze_dataset/constants.py`](/maze_dataset/constants.py)

- mypy doesn't recognize the fields in this dataclass  
  local link: [`/maze_dataset/constants.py:217`](/maze_dataset/constants.py#L217) 
  | view on GitHub: [maze_dataset/constants.py#L217](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/constants.py#L217)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=mypy%20doesn%27t%20recognize%20the%20fields%20in%20this%20dataclass&body=%23%20source%0A%0A%5B%60maze_dataset%2Fconstants.py%23L217%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fconstants.py%23L217%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20edit%20__getitem__%20to%20add%20warning%20for%20accessing%20a%20RESERVE%20token%0A%0A%23%20HACK%3A%20mypy%20doesn%27t%20recognize%20the%20fields%20in%20this%20dataclass%0AVOCAB%3A%20_VOCAB_BASE%20%3D%20_VOCAB_BASE%28%29%20%20%23%20type%3A%20ignore%0A%22public%20access%20to%20universal%20vocabulary%20for%20%60MazeTokenizerModular%60%22%0A%60%60%60&labels=HACK)

  ```python
  # TODO: edit __getitem__ to add warning for accessing a RESERVE token

  # HACK: mypy doesn't recognize the fields in this dataclass
  VOCAB: _VOCAB_BASE = _VOCAB_BASE()  # type: ignore
  "public access to universal vocabulary for `MazeTokenizerModular`"
  ```




## [`maze_dataset/maze/lattice_maze.py`](/maze_dataset/maze/lattice_maze.py)

- I think this is fine, but im not sure  
  local link: [`/maze_dataset/maze/lattice_maze.py:791`](/maze_dataset/maze/lattice_maze.py#L791) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L791](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L791)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=I%20think%20this%20is%20fine%2C%20but%20im%20not%20sure&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L791%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L791%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%29%0A%09%09%09output_maze%20%3D%20SolvedMaze.from_targeted_lattice_maze%28%0A%09%09%09%09%23%20HACK%3A%20I%20think%20this%20is%20fine%2C%20but%20im%20not%20sure%0A%09%09%09%09targeted_lattice_maze%3Doutput_maze%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09solution%3Dsolution%2C%0A%60%60%60&labels=HACK)

  ```python
  )
  output_maze = SolvedMaze.from_targeted_lattice_maze(
  	# HACK: I think this is fine, but im not sure
  	targeted_lattice_maze=output_maze,  # type: ignore[arg-type]
  	solution=solution,
  ```


- type ignores here fine since we check the instance  
  local link: [`/maze_dataset/maze/lattice_maze.py:809`](/maze_dataset/maze/lattice_maze.py#L809) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L809](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L809)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=type%20ignores%20here%20fine%20since%20we%20check%20the%20instance&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L809%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L809%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09Only%20legacy%20tokenizers%20and%20their%20%60MazeTokenizerModular%60%20analogs%20are%20supported.%0A%09%09%22%22%22%0A%09%09%23%20HACK%3A%20type%20ignores%20here%20fine%20since%20we%20check%20the%20instance%0A%09%09if%20isinstance_by_type_name%28maze_tokenizer%2C%20%22TokenizationMode%22%29%3A%0A%09%09%09maze_tokenizer%20%3D%20maze_tokenizer.to_legacy_tokenizer%28%29%20%20%23%20type%3A%20ignore%5Bunion-attr%5D%0A%60%60%60&labels=HACK)

  ```python
  Only legacy tokenizers and their `MazeTokenizerModular` analogs are supported.
  """
  # HACK: type ignores here fine since we check the instance
  if isinstance_by_type_name(maze_tokenizer, "TokenizationMode"):
  	maze_tokenizer = maze_tokenizer.to_legacy_tokenizer()  # type: ignore[union-attr]
  ```


- lots of `# type: ignore[attr-defined]` here since its defined for any `LatticeMaze`  
  local link: [`/maze_dataset/maze/lattice_maze.py:869`](/maze_dataset/maze/lattice_maze.py#L869) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L869](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L869)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=lots%20of%20%60%23%20type%3A%20ignore%5Battr-defined%5D%60%20here%20since%20its%20defined%20for%20any%20%60LatticeMaze%60&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L869%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L869%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09-%20used%20in%20%60RasterizedMazeDataset%60%2C%20which%20mimics%20the%20mazes%20in%20https%3A%2F%2Fgithub.com%2Faks2203%2Feasy-to-hard-data%0A%09%09%22%22%22%0A%09%09%23%20HACK%3A%20lots%20of%20%60%23%20type%3A%20ignore%5Battr-defined%5D%60%20here%20since%20its%20defined%20for%20any%20%60LatticeMaze%60%0A%09%09%23%20but%20solution%2C%20start_pos%2C%20end_pos%20not%20always%20defined%0A%09%09%23%20but%20its%20fine%20since%20we%20explicitly%20check%20the%20type%0A%60%60%60&labels=HACK)

  ```python
  - used in `RasterizedMazeDataset`, which mimics the mazes in https://github.com/aks2203/easy-to-hard-data
  """
  # HACK: lots of `# type: ignore[attr-defined]` here since its defined for any `LatticeMaze`
  # but solution, start_pos, end_pos not always defined
  # but its fine since we explicitly check the type
  ```


- idk why type ignore here  
  local link: [`/maze_dataset/maze/lattice_maze.py:1421`](/maze_dataset/maze/lattice_maze.py#L1421) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L1421](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L1421)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=idk%20why%20type%20ignore%20here&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L1421%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L1421%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%22%22%22%0A%09%09forks_idxs%2C%20_%20%3D%20self.get_solution_forking_points%28%29%0A%09%09%23%20HACK%3A%20idk%20why%20type%20ignore%20here%0A%09%09return%20%28%20%20%23%20type%3A%20ignore%5Breturn-value%5D%0A%09%09%09np.delete%28np.arange%28self.solution.shape%5B0%5D%29%2C%20forks_idxs%2C%20axis%3D0%29%2C%0A%60%60%60&labels=HACK)

  ```python
  """
  forks_idxs, _ = self.get_solution_forking_points()
  # HACK: idk why type ignore here
  return (  # type: ignore[return-value]
  	np.delete(np.arange(self.solution.shape[0]), forks_idxs, axis=0),
  ```




## [`maze_dataset/tokenization/modular/element_base.py`](/maze_dataset/tokenization/modular/element_base.py)

- this is not the right way of doing this lol  
  local link: [`/maze_dataset/tokenization/modular/element_base.py:296`](/maze_dataset/tokenization/modular/element_base.py#L296) 
  | view on GitHub: [maze_dataset/tokenization/modular/element_base.py#L296](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/element_base.py#L296)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20not%20the%20right%20way%20of%20doing%20this%20lol&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L296%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L296%29%0A%0A%23%20context%0A%60%60%60python%0A%09%22%22%22%0A%0A%09%23%20HACK%3A%20this%20is%20not%20the%20right%20way%20of%20doing%20this%20lol%0A%09key%3A%20str%20%3D%20NotImplementedError%20%20%23%20type%3A%20ignore%5Bassignment%5D%0A%60%60%60&labels=HACK)

  ```python
  """

  # HACK: this is not the right way of doing this lol
  key: str = NotImplementedError  # type: ignore[assignment]
  ```





# TODO

## [`maze_dataset/benchmark/config_sweep.py`](/maze_dataset/benchmark/config_sweep.py)

- B007 noqaed because we dont use `ep_kw_name` or `gf_idx`  
  local link: [`/maze_dataset/benchmark/config_sweep.py:408`](/maze_dataset/benchmark/config_sweep.py#L408) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L408](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L408)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=B007%20noqaed%20because%20we%20dont%20use%20%60ep_kw_name%60%20or%20%60gf_idx%60&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L408%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L408%29%0A%0A%23%20context%0A%60%60%60python%0A%09configs%3A%20list%5BMazeDatasetConfig%5D%20%3D%20list%28%29%0A%0A%09%23%20TODO%3A%20B007%20noqaed%20because%20we%20dont%20use%20%60ep_kw_name%60%20or%20%60gf_idx%60%0A%09for%20ep_kw_name%2C%20ep_kw%20in%20ep_kwargs%3A%20%20%23%20noqa%3A%20B007%0A%09%09for%20gf_idx%2C%20gen_func%20in%20enumerate%28generators%29%3A%20%20%23%20noqa%3A%20B007%0A%60%60%60&labels=enhancement)

  ```python
  configs: list[MazeDatasetConfig] = list()

  # TODO: B007 noqaed because we dont use `ep_kw_name` or `gf_idx`
  for ep_kw_name, ep_kw in ep_kwargs:  # noqa: B007
  	for gf_idx, gen_func in enumerate(generators):  # noqa: B007
  ```




## [`maze_dataset/constants.py`](/maze_dataset/constants.py)

- edit __getitem__ to add warning for accessing a RESERVE token  
  local link: [`/maze_dataset/constants.py:215`](/maze_dataset/constants.py#L215) 
  | view on GitHub: [maze_dataset/constants.py#L215](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/constants.py#L215)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=edit%20__getitem__%20to%20add%20warning%20for%20accessing%20a%20RESERVE%20token&body=%23%20source%0A%0A%5B%60maze_dataset%2Fconstants.py%23L215%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fconstants.py%23L215%29%0A%0A%23%20context%0A%60%60%60python%0A%29%0A%22combined%20vocab%20class%2C%20private%22%0A%23%20TODO%3A%20edit%20__getitem__%20to%20add%20warning%20for%20accessing%20a%20RESERVE%20token%0A%0A%23%20HACK%3A%20mypy%20doesn%27t%20recognize%20the%20fields%20in%20this%20dataclass%0A%60%60%60&labels=enhancement)

  ```python
  )
  "combined vocab class, private"
  # TODO: edit __getitem__ to add warning for accessing a RESERVE token

  # HACK: mypy doesn't recognize the fields in this dataclass
  ```




## [`maze_dataset/dataset/collected_dataset.py`](/maze_dataset/dataset/collected_dataset.py)

- remove duplication with MazeDatasetConfig().as_tokens() somehow?  
  local link: [`/maze_dataset/dataset/collected_dataset.py:191`](/maze_dataset/dataset/collected_dataset.py#L191) 
  | view on GitHub: [maze_dataset/dataset/collected_dataset.py#L191](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/collected_dataset.py#L191)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=remove%20duplication%20with%20MazeDatasetConfig%28%29.as_tokens%28%29%20somehow%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fcollected_dataset.py%23L191%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fcollected_dataset.py%23L191%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%29%0A%0A%09%23%20TODO%3A%20remove%20duplication%20with%20MazeDatasetConfig%28%29.as_tokens%28%29%20somehow%3F%0A%09def%20as_tokens%28%0A%09%09self%2C%0A%60%60%60&labels=enhancement)

  ```python
  	)

  # TODO: remove duplication with MazeDatasetConfig().as_tokens() somehow?
  def as_tokens(
  	self,
  ```


- MazeTokenizer  
  local link: [`/maze_dataset/dataset/collected_dataset.py:194`](/maze_dataset/dataset/collected_dataset.py#L194) 
  | view on GitHub: [maze_dataset/dataset/collected_dataset.py#L194](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/collected_dataset.py#L194)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=MazeTokenizer&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fcollected_dataset.py%23L194%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fcollected_dataset.py%23L194%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20as_tokens%28%0A%09%09self%2C%0A%09%09%23%20TODO%3A%20MazeTokenizer%0A%09%09maze_tokenizer%2C%20%20%23%20noqa%3A%20ANN001%0A%09%09limit%3A%20int%20%7C%20None%20%3D%20None%2C%0A%60%60%60&labels=enhancement)

  ```python
  def as_tokens(
  	self,
  	# TODO: MazeTokenizer
  	maze_tokenizer,  # noqa: ANN001
  	limit: int | None = None,
  ```


- why cant we set this directly? its not frozen, and it seems to work in a regular MazeDataset  
  local link: [`/maze_dataset/dataset/collected_dataset.py:219`](/maze_dataset/dataset/collected_dataset.py#L219) 
  | view on GitHub: [maze_dataset/dataset/collected_dataset.py#L219](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/collected_dataset.py#L219)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=why%20cant%20we%20set%20this%20directly%3F%20its%20not%20frozen%2C%20and%20it%20seems%20to%20work%20in%20a%20regular%20MazeDataset&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fcollected_dataset.py%23L219%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fcollected_dataset.py%23L219%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20update_self_config%28self%29%20-%3E%20None%3A%0A%09%09%22update%20the%20config%20to%20match%20the%20number%20of%20mazes%2C%20and%20update%20the%20underlying%20configs%20of%20each%20dataset%22%0A%09%09%23%20TODO%3A%20why%20cant%20we%20set%20this%20directly%3F%20its%20not%20frozen%2C%20and%20it%20seems%20to%20work%20in%20a%20regular%20MazeDataset%0A%09%09self.cfg.__dict__%5B%22n_mazes%22%5D%20%3D%20len%28self%29%0A%09%09for%20dataset%20in%20self.maze_datasets%3A%0A%60%60%60&labels=enhancement)

  ```python
  def update_self_config(self) -> None:
  	"update the config to match the number of mazes, and update the underlying configs of each dataset"
  	# TODO: why cant we set this directly? its not frozen, and it seems to work in a regular MazeDataset
  	self.cfg.__dict__["n_mazes"] = len(self)
  	for dataset in self.maze_datasets:
  ```




## [`maze_dataset/dataset/dataset.py`](/maze_dataset/dataset/dataset.py)

- get rid of all these things as part of migration to tokenizer-free dataset config  
  local link: [`/maze_dataset/dataset/dataset.py:70`](/maze_dataset/dataset/dataset.py#L70) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L70](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L70)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=get%20rid%20of%20all%20these%20things%20as%20part%20of%20migration%20to%20tokenizer-free%20dataset%20config&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L70%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L70%29%0A%0A%23%20context%0A%60%60%60python%0A%09name%3A%20str%0A%0A%09%23%20TODO%3A%20get%20rid%20of%20all%20these%20things%20as%20part%20of%20migration%20to%20tokenizer-free%20dataset%20config%0A%09%23%20--------------------------------------------------%0A%09seq_len_min%3A%20int%20%3D%20serializable_field%28default%3D1%29%0A%60%60%60&labels=enhancement)

  ```python
  name: str

  # TODO: get rid of all these things as part of migration to tokenizer-free dataset config
  # --------------------------------------------------
  seq_len_min: int = serializable_field(default=1)
  ```


- check the type here once muutils supports checking Callable signatures  
  local link: [`/maze_dataset/dataset/dataset.py:82`](/maze_dataset/dataset/dataset.py#L82) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L82](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L82)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=check%20the%20type%20here%20once%20muutils%20supports%20checking%20Callable%20signatures&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L82%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L82%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09default_factory%3Dlist%2C%0A%09%09deserialize_fn%3D_load_applied_filters%2C%0A%09%09assert_type%3DFalse%2C%20%20%23%20TODO%3A%20check%20the%20type%20here%20once%20muutils%20supports%20checking%20Callable%20signatures%0A%09%29%0A%60%60%60&labels=enhancement)

  ```python
  	default_factory=list,
  	deserialize_fn=_load_applied_filters,
  	assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
  )
  ```


- something here is broken  
  local link: [`/maze_dataset/dataset/dataset.py:92`](/maze_dataset/dataset/dataset.py#L92) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L92](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L92)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=something%20here%20is%20broken&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L92%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L92%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09self.seed%20%3D%20np.random.randint%282%2A%2A31%29%0A%0A%09%09%23%20TODO%3A%20something%20here%20is%20broken%0A%09%09if%20self.seed%20%21%3D%20GLOBAL_SEED%3A%0A%09%09%09warnings.warn%28%0A%60%60%60&labels=enhancement)

  ```python
  	self.seed = np.random.randint(2**31)

  # TODO: something here is broken
  if self.seed != GLOBAL_SEED:
  	warnings.warn(
  ```


- some funny business with manually specified filters here?  
  local link: [`/maze_dataset/dataset/dataset.py:435`](/maze_dataset/dataset/dataset.py#L435) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L435](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L435)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=some%20funny%20business%20with%20manually%20specified%20filters%20here%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L435%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L435%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20update%20the%20config%2C%20perform%20checks%0A%09%09%23%20TODO%3A%20some%20funny%20business%20with%20manually%20specified%20filters%20here%3F%0A%09%09output.update_self_config%28%29%0A%09%09_check_filter_equality%28%0A%60%60%60&labels=enhancement)

  ```python
  # update the config, perform checks
  # TODO: some funny business with manually specified filters here?
  output.update_self_config()
  _check_filter_equality(
  ```


- what the heck do we mean by the above? why the question mark? it should be a copy right?  
  local link: [`/maze_dataset/dataset/dataset.py:532`](/maze_dataset/dataset/dataset.py#L532) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L532](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L532)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=what%20the%20heck%20do%20we%20mean%20by%20the%20above%3F%20why%20the%20question%20mark%3F%20it%20should%20be%20a%20copy%20right%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L532%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L532%29%0A%0A%23%20context%0A%60%60%60python%0A%09be%20sure%20to%20return%20a%20COPY%2C%20not%20the%20original%3F%0A%09%23%20TODO%3A%20what%20the%20heck%20do%20we%20mean%20by%20the%20above%3F%20why%20the%20question%20mark%3F%20it%20should%20be%20a%20copy%20right%3F%0A%0A%09method%20should%20be%20a%20staticmethod%20of%20a%20namespace%20class%20registered%20with%20%60register_filter_namespace_for_dataset%60%0A%60%60%60&labels=enhancement)

  ```python
  be sure to return a COPY, not the original?
  # TODO: what the heck do we mean by the above? why the question mark? it should be a copy right?

  method should be a staticmethod of a namespace class registered with `register_filter_namespace_for_dataset`
  ```




## [`maze_dataset/dataset/filters.py`](/maze_dataset/dataset/filters.py)

- check for overlap?  
  local link: [`/maze_dataset/dataset/filters.py:118`](/maze_dataset/dataset/filters.py#L118) 
  | view on GitHub: [maze_dataset/dataset/filters.py#L118](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/filters.py#L118)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=check%20for%20overlap%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Ffilters.py%23L118%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Ffilters.py%23L118%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09-%20if%20two%20solutions%20are%20of%20different%20lengths%2C%20they%20will%20never%20be%20considered%20duplicates%0A%0A%09%09TODO%3A%20check%20for%20overlap%3F%0A%09%09%22%22%22%0A%09%09if%20len%28dataset%29%20%3E%20_max_dataset_len_threshold%3A%0A%60%60%60&labels=enhancement)

  ```python
  - if two solutions are of different lengths, they will never be considered duplicates

  TODO: check for overlap?
  """
  if len(dataset) > _max_dataset_len_threshold:
  ```


- `for` loop variable `value` overwritten by assignment target (Ruff PLW2901)  
  local link: [`/maze_dataset/dataset/filters.py:254`](/maze_dataset/dataset/filters.py#L254) 
  | view on GitHub: [maze_dataset/dataset/filters.py#L254](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/filters.py#L254)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=%60for%60%20loop%20variable%20%60value%60%20overwritten%20by%20assignment%20target%20%28Ruff%20PLW2901%29&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Ffilters.py%23L254%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Ffilters.py%23L254%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09elif%20isinstance%28value%2C%20%28list%2C%20np.ndarray%29%29%3A%20%20%23%20noqa%3A%20UP038%0A%09%09%09%09%09if%20isinstance%28value%2C%20list%29%3A%0A%09%09%09%09%09%09%23%20TODO%3A%20%60for%60%20loop%20variable%20%60value%60%20overwritten%20by%20assignment%20target%20%28Ruff%20PLW2901%29%0A%09%09%09%09%09%09try%3A%0A%09%09%09%09%09%09%09value%20%3D%20np.array%28value%29%20%20%23%20noqa%3A%20PLW2901%0A%60%60%60&labels=enhancement)

  ```python
  elif isinstance(value, (list, np.ndarray)):  # noqa: UP038
  	if isinstance(value, list):
  		# TODO: `for` loop variable `value` overwritten by assignment target (Ruff PLW2901)
  		try:
  			value = np.array(value)  # noqa: PLW2901
  ```




## [`maze_dataset/dataset/maze_dataset.py`](/maze_dataset/dataset/maze_dataset.py)

- don't use this unless generating in parallel!  
  local link: [`/maze_dataset/dataset/maze_dataset.py:51`](/maze_dataset/dataset/maze_dataset.py#L51) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L51](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L51)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=don%27t%20use%20this%20unless%20generating%20in%20parallel%21&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L51%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L51%29%0A%0A%23%20context%0A%60%60%60python%0A%09%22%22%22%0A%09global%20_GLOBAL_WORKER_CONFIG%20%20%23%20noqa%3A%20PLW0602%0A%09%23%20TODO%3A%20don%27t%20use%20this%20unless%20generating%20in%20parallel%21%0A%09maze%3A%20LatticeMaze%20%3D%20_GLOBAL_WORKER_CONFIG.maze_ctor%28%0A%09%09grid_shape%3D_GLOBAL_WORKER_CONFIG.grid_shape_np%2C%0A%60%60%60&labels=enhancement)

  ```python
  """
  global _GLOBAL_WORKER_CONFIG  # noqa: PLW0602
  # TODO: don't use this unless generating in parallel!
  maze: LatticeMaze = _GLOBAL_WORKER_CONFIG.maze_ctor(
  	grid_shape=_GLOBAL_WORKER_CONFIG.grid_shape_np,
  ```


- dont use globals here!  
  local link: [`/maze_dataset/dataset/maze_dataset.py:87`](/maze_dataset/dataset/maze_dataset.py#L87) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L87](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L87)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=dont%20use%20globals%20here%21&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L87%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L87%29%0A%0A%23%20context%0A%60%60%60python%0A%09%22%22%22%0A%09%23%20TODO%3A%20dont%20use%20globals%20here%21%0A%09global%20_GLOBAL_WORKER_CONFIG%20%20%23%20noqa%3A%20PLW0603%0A%09_GLOBAL_WORKER_CONFIG%20%3D%20config%0A%60%60%60&labels=enhancement)

  ```python
  """
  # TODO: dont use globals here!
  global _GLOBAL_WORKER_CONFIG  # noqa: PLW0603
  _GLOBAL_WORKER_CONFIG = config
  ```


- MazeTokenizer  
  local link: [`/maze_dataset/dataset/maze_dataset.py:205`](/maze_dataset/dataset/maze_dataset.py#L205) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L205](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L205)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=MazeTokenizer&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L205%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L205%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20as_tokens%28%0A%09%09self%2C%0A%09%09maze_tokenizer%2C%20%20%23%20TODO%3A%20MazeTokenizer%0A%09%09limit%3A%20int%20%7C%20None%20%3D%20None%2C%0A%09%09join_tokens_individual_maze%3A%20bool%20%3D%20False%2C%0A%60%60%60&labels=enhancement)

  ```python
  def as_tokens(
  	self,
  	maze_tokenizer,  # TODO: MazeTokenizer
  	limit: int | None = None,
  	join_tokens_individual_maze: bool = False,
  ```


- compare hashes of data instead of the data itself?  
  local link: [`/maze_dataset/dataset/maze_dataset.py:240`](/maze_dataset/dataset/maze_dataset.py#L240) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L240](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L240)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=compare%20hashes%20of%20data%20instead%20of%20the%20data%20itself%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L240%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L240%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%22can%20only%20compare%20with%20other%20MazeDataset%20objects%22%2C%0A%09%09%09%29%0A%09%09%23%20TODO%3A%20compare%20hashes%20of%20data%20instead%20of%20the%20data%20itself%3F%0A%09%09return%20self.cfg%20%3D%3D%20other.cfg%20and%20self.mazes%20%3D%3D%20other.mazes%0A%60%60%60&labels=enhancement)

  ```python
  		"can only compare with other MazeDataset objects",
  	)
  # TODO: compare hashes of data instead of the data itself?
  return self.cfg == other.cfg and self.mazes == other.mazes
  ```


- what to do when unexpected kwargs are passed?  
  local link: [`/maze_dataset/dataset/maze_dataset.py:256`](/maze_dataset/dataset/maze_dataset.py#L256) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L256](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L256)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=what%20to%20do%20when%20unexpected%20kwargs%20are%20passed%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L256%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L256%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09pool_kwargs%3A%20dict%20%7C%20None%20%3D%20None%2C%0A%09%09verbose%3A%20bool%20%3D%20False%2C%0A%09%09%23%20TODO%3A%20what%20to%20do%20when%20unexpected%20kwargs%20are%20passed%3F%0A%09%09%2A%2Akwargs%2C%20%20%23%20noqa%3A%20ARG003%0A%09%29%20-%3E%20%22MazeDataset%22%3A%0A%60%60%60&labels=enhancement)

  ```python
  	pool_kwargs: dict | None = None,
  	verbose: bool = False,
  	# TODO: what to do when unexpected kwargs are passed?
  	**kwargs,  # noqa: ARG003
  ) -> "MazeDataset":
  ```


- don't use the global unless generating in parallel!  
  local link: [`/maze_dataset/dataset/maze_dataset.py:277`](/maze_dataset/dataset/maze_dataset.py#L277) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L277](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L277)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=don%27t%20use%20the%20global%20unless%20generating%20in%20parallel%21&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L277%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L277%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09disable%3Dnot%20verbose%2C%0A%09%09%29%0A%09%09%23%20TODO%3A%20don%27t%20use%20the%20global%20unless%20generating%20in%20parallel%21%0A%09%09if%20gen_parallel%3A%0A%09%09%09with%20multiprocessing.Pool%28%0A%60%60%60&labels=enhancement)

  ```python
  	disable=not verbose,
  )
  # TODO: don't use the global unless generating in parallel!
  if gen_parallel:
  	with multiprocessing.Pool(
  ```


- the code below is for doing some smarter collecting and type checking. Probably will delete.  
  local link: [`/maze_dataset/dataset/maze_dataset.py:588`](/maze_dataset/dataset/maze_dataset.py#L588) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L588](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L588)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=the%20code%20below%20is%20for%20doing%20some%20smarter%20collecting%20and%20type%20checking.%20Probably%20will%20delete.&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L588%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L588%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20the%20code%20below%20is%20for%20doing%20some%20smarter%20collecting%20and%20type%20checking.%20Probably%20will%20delete.%0A%22%22%22%0Acollect%20either%20the%20type%20at%20the%20field%2C%20or%20the%20shape%20of%20the%20field%20if%20it%20is%20an%20array%0A%60%60%60&labels=enhancement)

  ```python
  # TODO: the code below is for doing some smarter collecting and type checking. Probably will delete.
  """
  collect either the type at the field, or the shape of the field if it is an array
  ```


- throw except here?  
  local link: [`/maze_dataset/dataset/maze_dataset.py:619`](/maze_dataset/dataset/maze_dataset.py#L619) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L619](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L619)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=throw%20except%20here%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L619%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L619%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09else%3A%0A%09%09%09%23%20its%20a%20list%20of%20something%20else%2C%20do%20a%20counter%20on%20those%0A%09%09%09%23%20TODO%3A%20throw%20except%20here%3F%0A%09%09%09metadata_actions%5Bkey%5D%20%3D%20Counter%0A%60%60%60&labels=enhancement)

  ```python
  else:
  	# its a list of something else, do a counter on those
  	# TODO: throw except here?
  	metadata_actions[key] = Counter
  ```


- throw except here?  
  local link: [`/maze_dataset/dataset/maze_dataset.py:630`](/maze_dataset/dataset/maze_dataset.py#L630) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L630](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L630)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=throw%20except%20here%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L630%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L630%29%0A%0A%23%20context%0A%60%60%60python%0A%09else%3A%0A%09%09%23%20counter%20for%20everything%20else%0A%09%09%23%20TODO%3A%20throw%20except%20here%3F%0A%09%09metadata_actions%5Bkey%5D%20%3D%20Counter%0A%22%22%22%0A%60%60%60&labels=enhancement)

  ```python
  	else:
  		# counter for everything else
  		# TODO: throw except here?
  		metadata_actions[key] = Counter
  """
  ```




## [`maze_dataset/dataset/maze_dataset_config.py`](/maze_dataset/dataset/maze_dataset_config.py)

- check the type here once muutils supports checking Callable signatures  
  local link: [`/maze_dataset/dataset/maze_dataset_config.py:163`](/maze_dataset/dataset/maze_dataset_config.py#L163) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset_config.py#L163](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset_config.py#L163)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=check%20the%20type%20here%20once%20muutils%20supports%20checking%20Callable%20signatures&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset_config.py%23L163%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset_config.py%23L163%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%7D%2C%0A%09%09loading_fn%3Dlambda%20data%3A%20_load_maze_ctor%28data%5B%22maze_ctor%22%5D%29%2C%0A%09%09assert_type%3DFalse%2C%20%20%23%20TODO%3A%20check%20the%20type%20here%20once%20muutils%20supports%20checking%20Callable%20signatures%0A%09%29%0A%60%60%60&labels=enhancement)

  ```python
  	},
  	loading_fn=lambda data: _load_maze_ctor(data["maze_ctor"]),
  	assert_type=False,  # TODO: check the type here once muutils supports checking Callable signatures
  )
  ```




## [`maze_dataset/maze/lattice_maze.py`](/maze_dataset/maze/lattice_maze.py)

- other connected components?  
  local link: [`/maze_dataset/maze/lattice_maze.py:375`](/maze_dataset/maze/lattice_maze.py#L375) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L375](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L375)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=other%20connected%20components%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L375%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L375%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%22%22%22get%20the%20largest%20%28and%20assumed%20only%20nonsingular%29%20connected%20component%20of%20the%20maze%0A%0A%09%09TODO%3A%20other%20connected%20components%3F%0A%09%09%22%22%22%0A%09%09if%20%28self.generation_meta%20is%20None%29%20or%20%28%0A%60%60%60&labels=enhancement)

  ```python
  """get the largest (and assumed only nonsingular) connected component of the maze

  TODO: other connected components?
  """
  if (self.generation_meta is None) or (
  ```


- dynamically generate visited_cells?  
  local link: [`/maze_dataset/maze/lattice_maze.py:389`](/maze_dataset/maze/lattice_maze.py#L389) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L389](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L389)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=dynamically%20generate%20visited_cells%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L389%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L389%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%29%0A%09%09%09if%20visited_cells%20is%20None%3A%0A%09%09%09%09%23%20TODO%3A%20dynamically%20generate%20visited_cells%3F%0A%09%09%09%09err_msg%3A%20str%20%3D%20f%22a%20maze%20which%20is%20not%20marked%20as%20fully%20connected%20must%20have%20a%20visited_cells%20field%20in%20its%20generation_meta%3A%20%7Bself.generation_meta%7D%5Cn%7Bself%7D%5Cn%7Bself.as_ascii%28%29%7D%22%0A%09%09%09%09raise%20ValueError%28%0A%60%60%60&labels=enhancement)

  ```python
  )
  if visited_cells is None:
  	# TODO: dynamically generate visited_cells?
  	err_msg: str = f"a maze which is not marked as fully connected must have a visited_cells field in its generation_meta: {self.generation_meta}\n{self}\n{self.as_ascii()}"
  	raise ValueError(
  ```


- any way to get return type hinting working for this?  
  local link: [`/maze_dataset/maze/lattice_maze.py:798`](/maze_dataset/maze/lattice_maze.py#L798) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L798](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L798)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=any%20way%20to%20get%20return%20type%20hinting%20working%20for%20this%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L798%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L798%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20output_maze%0A%0A%09%23%20TODO%3A%20any%20way%20to%20get%20return%20type%20hinting%20working%20for%20this%3F%0A%09%40classmethod%0A%09def%20from_tokens%28%0A%60%60%60&labels=enhancement)

  ```python
  	return output_maze

  # TODO: any way to get return type hinting working for this?
  @classmethod
  def from_tokens(
  ```


- make this less ugly  
  local link: [`/maze_dataset/maze/lattice_maze.py:1071`](/maze_dataset/maze/lattice_maze.py#L1071) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L1071](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L1071)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=make%20this%20less%20ugly&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L1071%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L1071%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%23%20use%20%60get_coord_neighbors%60%20to%20find%20connected%20neighbors%0A%09%09%09neighbors%3A%20CoordArray%20%3D%20temp_maze.get_coord_neighbors%28solution%5B-1%5D%29%0A%09%09%09%23%20TODO%3A%20make%20this%20less%20ugly%0A%09%09%09assert%20%28len%28neighbors.shape%29%20%3D%3D%202%29%20and%20%28neighbors.shape%5B1%5D%20%3D%3D%202%29%2C%20%28%20%20%23%20noqa%3A%20PT018%2C%20PLR2004%0A%09%09%09%09f%22neighbors%20%7Bneighbors%7D%20has%20shape%20%7Bneighbors.shape%7D%2C%20expected%20shape%20%28n%2C%202%29%5Cn%7Bneighbors%20%3D%20%7D%5Cn%7Bsolution%20%3D%20%7D%5Cn%7Bsolution_raw%20%3D%20%7D%5Cn%7Btemp_maze.as_ascii%28%29%7D%22%0A%60%60%60&labels=enhancement)

  ```python
  # use `get_coord_neighbors` to find connected neighbors
  neighbors: CoordArray = temp_maze.get_coord_neighbors(solution[-1])
  # TODO: make this less ugly
  assert (len(neighbors.shape) == 2) and (neighbors.shape[1] == 2), (  # noqa: PT018, PLR2004
  	f"neighbors {neighbors} has shape {neighbors.shape}, expected shape (n, 2)\n{neighbors = }\n{solution = }\n{solution_raw = }\n{temp_maze.as_ascii()}"
  ```


- the argument type is stricter than the expected type but it still fails?  
  local link: [`/maze_dataset/maze/lattice_maze.py:1304`](/maze_dataset/maze/lattice_maze.py#L1304) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L1304](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L1304)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=the%20argument%20type%20is%20stricter%20than%20the%20expected%20type%20but%20it%20still%20fails%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L1304%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L1304%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09connection_list%3Dconnection_list%2C%0A%09%09%09generation_meta%3Dgeneration_meta%2C%0A%09%09%09%23%20TODO%3A%20the%20argument%20type%20is%20stricter%20than%20the%20expected%20type%20but%20it%20still%20fails%3F%0A%09%09%09%23%20error%3A%20Argument%20%22start_pos%22%20to%20%22__init__%22%20of%20%22TargetedLatticeMaze%22%20has%20incompatible%20type%0A%09%09%09%23%20%22ndarray%5Btuple%5Bint%2C%20...%5D%2C%20dtype%5BAny%5D%5D%20%7C%20None%22%3B%20expected%20%22ndarray%5BAny%2C%20Any%5D%22%20%20%5Barg-type%5D%0A%60%60%60&labels=enhancement)

  ```python
  connection_list=connection_list,
  generation_meta=generation_meta,
  # TODO: the argument type is stricter than the expected type but it still fails?
  # error: Argument "start_pos" to "__init__" of "TargetedLatticeMaze" has incompatible type
  # "ndarray[tuple[int, ...], dtype[Any]] | None"; expected "ndarray[Any, Any]"  [arg-type]
  ```


- assert the path does not backtrack, walk through walls, etc?  
  local link: [`/maze_dataset/maze/lattice_maze.py:1323`](/maze_dataset/maze/lattice_maze.py#L1323) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L1323](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L1323)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=assert%20the%20path%20does%20not%20backtrack%2C%20walk%20through%20walls%2C%20etc%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L1323%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L1323%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%09f%22when%20trying%20to%20create%20a%20SolvedMaze%2C%20the%20given%20end_pos%20does%20not%20match%20the%20one%20in%20the%20solution%3A%20given%3D%7Bend_pos%7D%2C%20solution%3D%7Bself.end_pos%7D%22%0A%09%09%09%09%29%0A%09%09%09%23%20TODO%3A%20assert%20the%20path%20does%20not%20backtrack%2C%20walk%20through%20walls%2C%20etc%3F%0A%0A%09def%20__eq__%28self%2C%20other%3A%20object%29%20-%3E%20bool%3A%0A%60%60%60&labels=enhancement)

  ```python
  				f"when trying to create a SolvedMaze, the given end_pos does not match the one in the solution: given={end_pos}, solution={self.end_pos}"
  			)
  		# TODO: assert the path does not backtrack, walk through walls, etc?

  def __eq__(self, other: object) -> bool:
  ```


- figure out why this function doesnt work, or maybe just get rid of it  
  local link: [`/maze_dataset/maze/lattice_maze.py:1492`](/maze_dataset/maze/lattice_maze.py#L1492) 
  | view on GitHub: [maze_dataset/maze/lattice_maze.py#L1492](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/maze/lattice_maze.py#L1492)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=figure%20out%20why%20this%20function%20doesnt%20work%2C%20or%20maybe%20just%20get%20rid%20of%20it&body=%23%20source%0A%0A%5B%60maze_dataset%2Fmaze%2Flattice_maze.py%23L1492%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fmaze%2Flattice_maze.py%23L1492%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20figure%20out%20why%20this%20function%20doesnt%20work%2C%20or%20maybe%20just%20get%20rid%20of%20it%0A%23%20def%20_remove_isolated_cells_old%28%0A%23%20%20%20%20%20image%3A%20Int%5Bnp.ndarray%2C%20%22RGB%20x%20y%22%5D%2C%0A%60%60%60&labels=enhancement)

  ```python
  # TODO: figure out why this function doesnt work, or maybe just get rid of it
  # def _remove_isolated_cells_old(
  #     image: Int[np.ndarray, "RGB x y"],
  ```




## [`maze_dataset/plotting/plot_maze.py`](/maze_dataset/plotting/plot_maze.py)

- this is a hack, we make the walls black (while still allowing negative values) by setting the nan color to black  
  local link: [`/maze_dataset/plotting/plot_maze.py:398`](/maze_dataset/plotting/plot_maze.py#L398) 
  | view on GitHub: [maze_dataset/plotting/plot_maze.py#L398](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/plot_maze.py#L398)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20a%20hack%2C%20we%20make%20the%20walls%20black%20%28while%20still%20allowing%20negative%20values%29%20by%20setting%20the%20nan%20color%20to%20black&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fplot_maze.py%23L398%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fplot_maze.py%23L398%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%23%20create%20colormap%0A%09%09%09cmap%20%3D%20mpl.colormaps%5Bself.node_color_map%5D%0A%09%09%09%23%20TODO%3A%20this%20is%20a%20hack%2C%20we%20make%20the%20walls%20black%20%28while%20still%20allowing%20negative%20values%29%20by%20setting%20the%20nan%20color%20to%20black%0A%09%09%09cmap.set_bad%28color%3D%22black%22%29%0A%60%60%60&labels=enhancement)

  ```python
  # create colormap
  cmap = mpl.colormaps[self.node_color_map]
  # TODO: this is a hack, we make the walls black (while still allowing negative values) by setting the nan color to black
  cmap.set_bad(color="black")
  ```


- this is a hack, but if you add 1 always then non-node valued plots have their walls dissapear. if you dont add 1, you get ugly colors between nodes when they are colored  
  local link: [`/maze_dataset/plotting/plot_maze.py:475`](/maze_dataset/plotting/plot_maze.py#L475) 
  | view on GitHub: [maze_dataset/plotting/plot_maze.py#L475](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/plot_maze.py#L475)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20a%20hack%2C%20but%20if%20you%20add%201%20always%20then%20non-node%20valued%20plots%20have%20their%20walls%20dissapear.%20if%20you%20dont%20add%201%2C%20you%20get%20ugly%20colors%20between%20nodes%20when%20they%20are%20colored&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fplot_maze.py%23L475%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fplot_maze.py%23L475%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09Returns%20a%20matrix%20of%20side%20length%20%28ul%29%20%2A%20n%20%2B%201%20where%20n%20is%20the%20number%20of%20nodes.%0A%09%09%22%22%22%0A%09%09%23%20TODO%3A%20this%20is%20a%20hack%2C%20but%20if%20you%20add%201%20always%20then%20non-node%20valued%20plots%20have%20their%20walls%20dissapear.%20if%20you%20dont%20add%201%2C%20you%20get%20ugly%20colors%20between%20nodes%20when%20they%20are%20colored%0A%09%09node_bdry_hack%3A%20int%0A%09%09connection_list_processed%3A%20Float%5Bnp.ndarray%2C%20%22dim%20row%20col%22%5D%0A%60%60%60&labels=enhancement)

  ```python
  Returns a matrix of side length (ul) * n + 1 where n is the number of nodes.
  """
  # TODO: this is a hack, but if you add 1 always then non-node valued plots have their walls dissapear. if you dont add 1, you get ugly colors between nodes when they are colored
  node_bdry_hack: int
  connection_list_processed: Float[np.ndarray, "dim row col"]
  ```


- hack  
  local link: [`/maze_dataset/plotting/plot_maze.py:483`](/maze_dataset/plotting/plot_maze.py#L483) 
  | view on GitHub: [maze_dataset/plotting/plot_maze.py#L483](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/plot_maze.py#L483)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=hack&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fplot_maze.py%23L483%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fplot_maze.py%23L483%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09connection_values%20%3D%20scaled_node_values%20%2A%20connection_val_scale%0A%09%09%09node_bdry_hack%20%3D%200%0A%09%09%09%23%20TODO%3A%20hack%0A%09%09%09%23%20invert%20connection%20list%0A%09%09%09connection_list_processed%20%3D%20np.logical_not%28self.maze.connection_list%29%0A%60%60%60&labels=enhancement)

  ```python
  connection_values = scaled_node_values * connection_val_scale
  node_bdry_hack = 0
  # TODO: hack
  # invert connection list
  connection_list_processed = np.logical_not(self.maze.connection_list)
  ```


- hack  
  local link: [`/maze_dataset/plotting/plot_maze.py:487`](/maze_dataset/plotting/plot_maze.py#L487) 
  | view on GitHub: [maze_dataset/plotting/plot_maze.py#L487](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/plot_maze.py#L487)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=hack&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fplot_maze.py%23L487%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fplot_maze.py%23L487%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09connection_list_processed%20%3D%20np.logical_not%28self.maze.connection_list%29%0A%09%09else%3A%0A%09%09%09%23%20TODO%3A%20hack%0A%09%09%09scaled_node_values%20%3D%20self.node_values%0A%09%09%09%23%20connection_values%20%3D%20scaled_node_values%0A%60%60%60&labels=enhancement)

  ```python
  	connection_list_processed = np.logical_not(self.maze.connection_list)
  else:
  	# TODO: hack
  	scaled_node_values = self.node_values
  	# connection_values = scaled_node_values
  ```




## [`maze_dataset/plotting/print_tokens.py`](/maze_dataset/plotting/print_tokens.py)

- why are we using a map here again?  
  local link: [`/maze_dataset/plotting/print_tokens.py:89`](/maze_dataset/plotting/print_tokens.py#L89) 
  | view on GitHub: [maze_dataset/plotting/print_tokens.py#L89](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/print_tokens.py#L89)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=why%20are%20we%20using%20a%20map%20here%20again%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fprint_tokens.py%23L89%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fprint_tokens.py%23L89%29%0A%0A%23%20context%0A%60%60%60python%0A%09if%20max_length%20is%20not%20None%3A%0A%09%09%23%20TODO%3A%20why%20are%20we%20using%20a%20map%20here%20again%3F%0A%09%09%23%20TYPING%3A%20this%20is%20missing%20a%20lot%20of%20type%20hints%0A%09%09wrapped%3A%20list%20%3D%20list%28%20%20%23%20noqa%3A%20C417%0A%60%60%60&labels=enhancement)

  ```python
  if max_length is not None:
  	# TODO: why are we using a map here again?
  	# TYPING: this is missing a lot of type hints
  	wrapped: list = list(  # noqa: C417
  ```




## [`maze_dataset/tokenization/maze_tokenizer_legacy.py`](/maze_dataset/tokenization/maze_tokenizer_legacy.py)

- this is hacky, but we don't want to modify the original SPECIAL_TOKENS since that will break old models  
  local link: [`/maze_dataset/tokenization/maze_tokenizer_legacy.py:257`](/maze_dataset/tokenization/maze_tokenizer_legacy.py#L257) 
  | view on GitHub: [maze_dataset/tokenization/maze_tokenizer_legacy.py#L257](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/maze_tokenizer_legacy.py#L257)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20hacky%2C%20but%20we%20don%27t%20want%20to%20modify%20the%20original%20SPECIAL_TOKENS%20since%20that%20will%20break%20old%20models&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmaze_tokenizer_legacy.py%23L257%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmaze_tokenizer_legacy.py%23L257%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%29%0A%09%09elif%20self.tokenization_mode%20%3D%3D%20TokenizationMode.AOTP_CTT_indexed%3A%0A%09%09%09%23%20TODO%3A%20this%20is%20hacky%2C%20but%20we%20don%27t%20want%20to%20modify%20the%20original%20SPECIAL_TOKENS%20since%20that%20will%20break%20old%20models%0A%09%09%09output.extend%28%0A%09%09%09%09%5B%0A%60%60%60&labels=enhancement)

  ```python
  	)
  elif self.tokenization_mode == TokenizationMode.AOTP_CTT_indexed:
  	# TODO: this is hacky, but we don't want to modify the original SPECIAL_TOKENS since that will break old models
  	output.extend(
  		[
  ```


- deprecate  
  local link: [`/maze_dataset/tokenization/maze_tokenizer_legacy.py:306`](/maze_dataset/tokenization/maze_tokenizer_legacy.py#L306) 
  | view on GitHub: [maze_dataset/tokenization/maze_tokenizer_legacy.py#L306](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/maze_tokenizer_legacy.py#L306)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=deprecate&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmaze_tokenizer_legacy.py%23L306%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmaze_tokenizer_legacy.py%23L306%29%0A%0A%23%20context%0A%60%60%60python%0A%09%40property%0A%09def%20_n_tokens%28self%29%20-%3E%20int%3A%0A%09%09%23%20TODO%3A%20deprecate%0A%09%09return%20self._vocab_size%0A%60%60%60&labels=enhancement)

  ```python
  @property
  def _n_tokens(self) -> int:
  	# TODO: deprecate
  	return self._vocab_size
  ```




## [`maze_dataset/tokenization/modular/all_instances.py`](/maze_dataset/tokenization/modular/all_instances.py)

- what is this magic value here exactly?  
  local link: [`/maze_dataset/tokenization/modular/all_instances.py:143`](/maze_dataset/tokenization/modular/all_instances.py#L143) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_instances.py#L143](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_instances.py#L143)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=what%20is%20this%20magic%20value%20here%20exactly%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L143%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L143%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09validation_funcs%3A%20frozendict.frozendict%0A%09%09%23%20TODO%3A%20what%20is%20this%20magic%20value%20here%20exactly%3F%0A%09%09if%20len%28args%29%20%3E%3D%202%20and%20args%5B1%5D%20is%20not%20None%3A%20%20%23%20noqa%3A%20PLR2004%0A%09%09%09validation_funcs%20%3D%20frozendict.frozendict%28args%5B1%5D%29%0A%60%60%60&labels=enhancement)

  ```python
  validation_funcs: frozendict.frozendict
  # TODO: what is this magic value here exactly?
  if len(args) >= 2 and args[1] is not None:  # noqa: PLR2004
  	validation_funcs = frozendict.frozendict(args[1])
  ```




## [`maze_dataset/tokenization/modular/all_tokenizers.py`](/maze_dataset/tokenization/modular/all_tokenizers.py)

- add more here as specific tokenizers become canonical and frequently used  
  local link: [`/maze_dataset/tokenization/modular/all_tokenizers.py:102`](/maze_dataset/tokenization/modular/all_tokenizers.py#L102) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_tokenizers.py#L102](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_tokenizers.py#L102)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=add%20more%20here%20as%20specific%20tokenizers%20become%20canonical%20and%20frequently%20used&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L102%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L102%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09prompt_sequencer%3DPromptSequencers.AOTP%28coord_tokenizer%3DCoordTokenizers.CTT%28%29%29%2C%0A%09%29%2C%0A%09%23%20TODO%3A%20add%20more%20here%20as%20specific%20tokenizers%20become%20canonical%20and%20frequently%20used%0A%5D%0A%60%60%60&labels=enhancement)

  ```python
  		prompt_sequencer=PromptSequencers.AOTP(coord_tokenizer=CoordTokenizers.CTT()),
  	),
  	# TODO: add more here as specific tokenizers become canonical and frequently used
  ]
  ```




## [`maze_dataset/tokenization/modular/element_base.py`](/maze_dataset/tokenization/modular/element_base.py)

- why noqa here? `B024 `__TokenizerElementNamespace` is an abstract base class, but it has no abstract methods or properties`  
  local link: [`/maze_dataset/tokenization/modular/element_base.py:288`](/maze_dataset/tokenization/modular/element_base.py#L288) 
  | view on GitHub: [maze_dataset/tokenization/modular/element_base.py#L288](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/element_base.py#L288)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=why%20noqa%20here%3F%20%60B024%20%60__TokenizerElementNamespace%60%20is%20an%20abstract%20base%20class%2C%20but%20it%20has%20no%20abstract%20methods%20or%20properties%60&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L288%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L288%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20why%20noqa%20here%3F%20%60B024%20%60__TokenizerElementNamespace%60%20is%20an%20abstract%20base%20class%2C%20but%20it%20has%20no%20abstract%20methods%20or%20properties%60%0Aclass%20__TokenizerElementNamespace%28abc.ABC%29%3A%20%20%23%20noqa%3A%20B024%0A%09%22%22%22ABC%20for%20namespaces%0A%60%60%60&labels=enhancement)

  ```python
  # TODO: why noqa here? `B024 `__TokenizerElementNamespace` is an abstract base class, but it has no abstract methods or properties`
  class __TokenizerElementNamespace(abc.ABC):  # noqa: B024
  	"""ABC for namespaces
  ```




## [`maze_dataset/tokenization/modular/elements.py`](/maze_dataset/tokenization/modular/elements.py)

- make this a static/class method, allowing ForksAndStraightaways to skip object construction at every call  
  local link: [`/maze_dataset/tokenization/modular/elements.py:680`](/maze_dataset/tokenization/modular/elements.py#L680) 
  | view on GitHub: [maze_dataset/tokenization/modular/elements.py#L680](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/elements.py#L680)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=make%20this%20a%20static%2Fclass%20method%2C%20allowing%20ForksAndStraightaways%20to%20skip%20object%20construction%20at%20every%20call&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felements.py%23L680%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felements.py%23L680%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09return%20StepSizes.key%0A%0A%09%09%40abc.abstractmethod%20%20%23%20TODO%3A%20make%20this%20a%20static%2Fclass%20method%2C%20allowing%20ForksAndStraightaways%20to%20skip%20object%20construction%20at%20every%20call%0A%09%09def%20_step_single_indices%28self%2C%20maze%3A%20SolvedMaze%29%20-%3E%20list%5Bint%5D%3A%0A%09%09%09%22%22%22Returns%20the%20indices%20of%20%60maze.solution%60%20corresponding%20to%20the%20steps%20to%20be%20tokenized.%22%22%22%0A%60%60%60&labels=enhancement)

  ```python
  	return StepSizes.key

  @abc.abstractmethod  # TODO: make this a static/class method, allowing ForksAndStraightaways to skip object construction at every call
  def _step_single_indices(self, maze: SolvedMaze) -> list[int]:
  	"""Returns the indices of `maze.solution` corresponding to the steps to be tokenized."""
  ```


- RUF007 Prefer `itertools.pairwise()` over `zip()` when iterating over successive pairs  
  local link: [`/maze_dataset/tokenization/modular/elements.py:690`](/maze_dataset/tokenization/modular/elements.py#L690) 
  | view on GitHub: [maze_dataset/tokenization/modular/elements.py#L690](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/elements.py#L690)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=RUF007%20Prefer%20%60itertools.pairwise%28%29%60%20over%20%60zip%28%29%60%20when%20iterating%20over%20successive%20pairs&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felements.py%23L690%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felements.py%23L690%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%22%22%22Returns%20steps%20as%20tuples%20of%20starting%20and%20ending%20positions%20for%20each%20step.%22%22%22%0A%09%09%09indices%3A%20list%5Bint%5D%20%3D%20self._step_single_indices%28maze%29%0A%09%09%09%23%20TODO%3A%20RUF007%20Prefer%20%60itertools.pairwise%28%29%60%20over%20%60zip%28%29%60%20when%20iterating%20over%20successive%20pairs%0A%09%09%09return%20%5B%0A%09%09%09%09%28start%2C%20end%29%0A%60%60%60&labels=enhancement)

  ```python
  """Returns steps as tuples of starting and ending positions for each step."""
  indices: list[int] = self._step_single_indices(maze)
  # TODO: RUF007 Prefer `itertools.pairwise()` over `zip()` when iterating over successive pairs
  return [
  	(start, end)
  ```




## [`maze_dataset/tokenization/modular/maze_tokenizer_modular.py`](/maze_dataset/tokenization/modular/maze_tokenizer_modular.py)

- unclear why we need to use `noqa: N805` here since its a classmethod  
  local link: [`/maze_dataset/tokenization/modular/maze_tokenizer_modular.py:291`](/maze_dataset/tokenization/modular/maze_tokenizer_modular.py#L291) 
  | view on GitHub: [maze_dataset/tokenization/modular/maze_tokenizer_modular.py#L291](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/maze_tokenizer_modular.py#L291)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=unclear%20why%20we%20need%20to%20use%20%60noqa%3A%20N805%60%20here%20since%20its%20a%20classmethod&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fmaze_tokenizer_modular.py%23L291%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fmaze_tokenizer_modular.py%23L291%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%29%0A%0A%09%23%20TODO%3A%20unclear%20why%20we%20need%20to%20use%20%60noqa%3A%20N805%60%20here%20since%20its%20a%20classmethod%0A%09%23%20maybe%20we%20need%20to%20hit%20every%20overload%20with%20%60%40classmethod%60%3F%0A%09%40overload%0A%60%60%60&labels=enhancement)

  ```python
  	)

  # TODO: unclear why we need to use `noqa: N805` here since its a classmethod
  # maybe we need to hit every overload with `@classmethod`?
  @overload
  ```




## [`tests/unit/dataset/test_collected_dataset.py`](/tests/unit/dataset/test_collected_dataset.py)

- test downloading after we implement downloading datasets  
  local link: [`/tests/unit/dataset/test_collected_dataset.py:83`](/tests/unit/dataset/test_collected_dataset.py#L83) 
  | view on GitHub: [tests/unit/dataset/test_collected_dataset.py#L83](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/dataset/test_collected_dataset.py#L83)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=test%20downloading%20after%20we%20implement%20downloading%20datasets&body=%23%20source%0A%0A%5B%60tests%2Funit%2Fdataset%2Ftest_collected_dataset.py%23L83%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Fdataset%2Ftest_collected_dataset.py%23L83%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20test_download%28self%29%3A%0A%09%09%23%20TODO%3A%20test%20downloading%20after%20we%20implement%20downloading%20datasets%0A%09%09pass%0A%60%60%60&labels=enhancement)

  ```python
  def test_download(self):
  	# TODO: test downloading after we implement downloading datasets
  	pass
  ```




## [`tests/unit/generation/test_coord_str_tuple.py`](/tests/unit/generation/test_coord_str_tuple.py)

- test for negative coords  
  local link: [`/tests/unit/generation/test_coord_str_tuple.py:23`](/tests/unit/generation/test_coord_str_tuple.py#L23) 
  | view on GitHub: [tests/unit/generation/test_coord_str_tuple.py#L23](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/generation/test_coord_str_tuple.py#L23)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=test%20for%20negative%20coords&body=%23%20source%0A%0A%5B%60tests%2Funit%2Fgeneration%2Ftest_coord_str_tuple.py%23L23%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Fgeneration%2Ftest_coord_str_tuple.py%23L23%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TODO%3A%20test%20for%20negative%20coords%0A%60%60%60&labels=enhancement)

  ```python
  # TODO: test for negative coords
  ```


- resolve testing duplication in test_token_utils.py  
  local link: [`/tests/unit/generation/test_coord_str_tuple.py:66`](/tests/unit/generation/test_coord_str_tuple.py#L66) 
  | view on GitHub: [tests/unit/generation/test_coord_str_tuple.py#L66](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/generation/test_coord_str_tuple.py#L66)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=resolve%20testing%20duplication%20in%20test_token_utils.py&body=%23%20source%0A%0A%5B%60tests%2Funit%2Fgeneration%2Ftest_coord_str_tuple.py%23L66%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Fgeneration%2Ftest_coord_str_tuple.py%23L66%29%0A%0A%23%20context%0A%60%60%60python%0Adef%20test_coords_to_strings%28%29%3A%0A%09%23%20TODO%3A%20resolve%20testing%20duplication%20in%20test_token_utils.py%0A%09assert%20coords_to_strings%28%0A%09%09%5B%281%2C%202%29%2C%20%22%3CADJLIST_START%3E%22%2C%20%285%2C%206%29%5D%2C%0A%60%60%60&labels=enhancement)

  ```python
  def test_coords_to_strings():
  	# TODO: resolve testing duplication in test_token_utils.py
  	assert coords_to_strings(
  		[(1, 2), "<ADJLIST_START>", (5, 6)],
  ```




## [`tests/unit/generation/test_maze_dataset.py`](/tests/unit/generation/test_maze_dataset.py)

- dataset.data_hash doesn't work right now  
  local link: [`/tests/unit/generation/test_maze_dataset.py:67`](/tests/unit/generation/test_maze_dataset.py#L67) 
  | view on GitHub: [tests/unit/generation/test_maze_dataset.py#L67](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/generation/test_maze_dataset.py#L67)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=dataset.data_hash%20doesn%27t%20work%20right%20now&body=%23%20source%0A%0A%5B%60tests%2Funit%2Fgeneration%2Ftest_maze_dataset.py%23L67%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Fgeneration%2Ftest_maze_dataset.py%23L67%29%0A%0A%23%20context%0A%60%60%60python%0Adef%20test_data_hash_wip%28%29%3A%0A%09dataset%20%3D%20MazeDataset.generate%28TEST_CONFIGS%5B0%5D%29%0A%09%23%20TODO%3A%20dataset.data_hash%20doesn%27t%20work%20right%20now%0A%09assert%20dataset%0A%60%60%60&labels=enhancement)

  ```python
  def test_data_hash_wip():
  	dataset = MazeDataset.generate(TEST_CONFIGS[0])
  	# TODO: dataset.data_hash doesn't work right now
  	assert dataset
  ```


- PERF401 Use `list.extend` to create a transformed list  
  local link: [`/tests/unit/generation/test_maze_dataset.py:318`](/tests/unit/generation/test_maze_dataset.py#L318) 
  | view on GitHub: [tests/unit/generation/test_maze_dataset.py#L318](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/generation/test_maze_dataset.py#L318)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=PERF401%20Use%20%60list.extend%60%20to%20create%20a%20transformed%20list&body=%23%20source%0A%0A%5B%60tests%2Funit%2Fgeneration%2Ftest_maze_dataset.py%23L318%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Fgeneration%2Ftest_maze_dataset.py%23L318%29%0A%0A%23%20context%0A%60%60%60python%0A%09mazes%3A%20list%5BSolvedMaze%5D%20%3D%20list%28%29%0A%09for%20maze_ascii%20in%20ascii_rep%3A%0A%09%09%23%20TODO%3A%20PERF401%20Use%20%60list.extend%60%20to%20create%20a%20transformed%20list%0A%09%09mazes.append%28SolvedMaze.from_ascii%28maze_ascii.strip%28%29%29%29%0A%60%60%60&labels=enhancement)

  ```python
  mazes: list[SolvedMaze] = list()
  for maze_ascii in ascii_rep:
  	# TODO: PERF401 Use `list.extend` to create a transformed list
  	mazes.append(SolvedMaze.from_ascii(maze_ascii.strip()))
  ```




## [`tests/unit/tokenization/test_maze_tokenization.py`](/tests/unit/tokenization/test_maze_tokenization.py)

- can't test that these match because order in adjacency list is random  
  local link: [`/tests/unit/tokenization/test_maze_tokenization.py:36`](/tests/unit/tokenization/test_maze_tokenization.py#L36) 
  | view on GitHub: [tests/unit/tokenization/test_maze_tokenization.py#L36](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/tokenization/test_maze_tokenization.py#L36)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=can%27t%20test%20that%20these%20match%20because%20order%20in%20adjacency%20list%20is%20random&body=%23%20source%0A%0A%5B%60tests%2Funit%2Ftokenization%2Ftest_maze_tokenization.py%23L36%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Ftokenization%2Ftest_maze_tokenization.py%23L36%29%0A%0A%23%20context%0A%60%60%60python%0A%09%23%20%29%0A%0A%09%23%20TODO%3A%20can%27t%20test%20that%20these%20match%20because%20order%20in%20adjacency%20list%20is%20random%0A%0A%09dataset_tokenized_individual%3A%20list%5Blist%5Bstr%5D%5D%20%3D%20%5B%0A%60%60%60&labels=enhancement)

  ```python
  # )

  # TODO: can't test that these match because order in adjacency list is random

  dataset_tokenized_individual: list[list[str]] = [
  ```





# TYPING

## [`maze_dataset/benchmark/config_sweep.py`](/maze_dataset/benchmark/config_sweep.py)

- error: Argument "func" to "run_maybe_parallel" has incompatible type "partial[list[SweepReturnType]]"; expected "Callable[[MazeDatasetConfig], float]"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:230`](/maze_dataset/benchmark/config_sweep.py#L230) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L230](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L230)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%20%22func%22%20to%20%22run_maybe_parallel%22%20has%20incompatible%20type%20%22partial%5Blist%5BSweepReturnType%5D%5D%22%3B%20expected%20%22Callable%5B%5BMazeDatasetConfig%5D%2C%20float%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L230%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L230%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09result_values_list%3A%20list%5Bfloat%5D%20%3D%20run_maybe_parallel%28%0A%09%09%09%23%20TYPING%3A%20error%3A%20Argument%20%22func%22%20to%20%22run_maybe_parallel%22%20has%20incompatible%20type%20%22partial%5Blist%5BSweepReturnType%5D%5D%22%3B%20expected%20%22Callable%5B%5BMazeDatasetConfig%5D%2C%20float%5D%22%20%20%5Barg-type%5D%0A%09%09%09func%3Dfunctools.partial%28%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09sweep%2C%0A%60%60%60&labels=TYPING)

  ```python
  result_values_list: list[float] = run_maybe_parallel(
  	# TYPING: error: Argument "func" to "run_maybe_parallel" has incompatible type "partial[list[SweepReturnType]]"; expected "Callable[[MazeDatasetConfig], float]"  [arg-type]
  	func=functools.partial(  # type: ignore[arg-type]
  		sweep,
  ```


- error: Argument "result_values" to "SweepResult" has incompatible type "dict[str, ndarray[Any, Any]]"; expected "dict[str, Sequence[SweepReturnType]]"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:250`](/maze_dataset/benchmark/config_sweep.py#L250) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L250](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L250)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%20%22result_values%22%20to%20%22SweepResult%22%20has%20incompatible%20type%20%22dict%5Bstr%2C%20ndarray%5BAny%2C%20Any%5D%5D%22%3B%20expected%20%22dict%5Bstr%2C%20Sequence%5BSweepReturnType%5D%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L250%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L250%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09configs%3Dconfigs%2C%0A%09%09%09param_values%3Dparam_values%2C%0A%09%09%09%23%20TYPING%3A%20error%3A%20Argument%20%22result_values%22%20to%20%22SweepResult%22%20has%20incompatible%20type%20%22dict%5Bstr%2C%20ndarray%5BAny%2C%20Any%5D%5D%22%3B%20expected%20%22dict%5Bstr%2C%20Sequence%5BSweepReturnType%5D%5D%22%20%20%5Barg-type%5D%0A%09%09%09result_values%3Dresult_values%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09param_key%3Dparam_key%2C%0A%60%60%60&labels=TYPING)

  ```python
  configs=configs,
  param_values=param_values,
  # TYPING: error: Argument "result_values" to "SweepResult" has incompatible type "dict[str, ndarray[Any, Any]]"; expected "dict[str, Sequence[SweepReturnType]]"  [arg-type]
  result_values=result_values,  # type: ignore[arg-type]
  param_key=param_key,
  ```


- error: Argument 1 to "plot" of "Axes" has incompatible type "list[ParamType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:291`](/maze_dataset/benchmark/config_sweep.py#L291) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L291](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L291)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22list%5BParamType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L291%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L291%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%29%3A%0A%09%09%09ax_.plot%28%0A%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22list%5BParamType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D%0A%09%09%09%09self.param_values%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%202%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22Sequence%5BSweepReturnType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  ):
  	ax_.plot(
  		# TYPING: error: Argument 1 to "plot" of "Axes" has incompatible type "list[ParamType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
  		self.param_values,  # type: ignore[arg-type]
  		# TYPING: error: Argument 2 to "plot" of "Axes" has incompatible type "Sequence[SweepReturnType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
  ```


- error: Argument 2 to "plot" of "Axes" has incompatible type "Sequence[SweepReturnType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:293`](/maze_dataset/benchmark/config_sweep.py#L293) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L293](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L293)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%202%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22Sequence%5BSweepReturnType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L293%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L293%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22list%5BParamType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D%0A%09%09%09%09self.param_values%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%202%20to%20%22plot%22%20of%20%22Axes%22%20has%20incompatible%20type%20%22Sequence%5BSweepReturnType%5D%22%3B%20expected%20%22float%20%7C%20Buffer%20%7C%20_SupportsArray%5Bdtype%5BAny%5D%5D%20%7C%20_NestedSequence%5B_SupportsArray%5Bdtype%5BAny%5D%5D%5D%20%7C%20bool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%20%7C%20_NestedSequence%5Bbool%20%7C%20int%20%7C%20float%20%7C%20complex%20%7C%20str%20%7C%20bytes%5D%20%7C%20str%22%20%20%5Barg-type%5D%0A%09%09%09%09result_values%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09%22.-%22%2C%0A%60%60%60&labels=TYPING)

  ```python
  # TYPING: error: Argument 1 to "plot" of "Axes" has incompatible type "list[ParamType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
  self.param_values,  # type: ignore[arg-type]
  # TYPING: error: Argument 2 to "plot" of "Axes" has incompatible type "Sequence[SweepReturnType]"; expected "float | Buffer | _SupportsArray[dtype[Any]] | _NestedSequence[_SupportsArray[dtype[Any]]] | bool | int | float | complex | str | bytes | _NestedSequence[bool | int | float | complex | str | bytes] | str"  [arg-type]
  result_values,  # type: ignore[arg-type]
  ".-",
  ```


- error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:316`](/maze_dataset/benchmark/config_sweep.py#L316) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L316](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L316)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%202%20to%20%22isinstance%22%20has%20incompatible%20type%20%22%3Ctyping%20special%20form%3E%22%3B%20expected%20%22_ClassInfo%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L316%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L316%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09%09%5B%0A%09%09%09%09%09%09f%22%7Bk%7D%3D%7Bcfg_shared%5Bk%5D.__name__%7D%22%0A%09%09%09%09%09%09%23%20TYPING%3A%20error%3A%20Argument%202%20to%20%22isinstance%22%20has%20incompatible%20type%20%22%3Ctyping%20special%20form%3E%22%3B%20expected%20%22_ClassInfo%22%20%20%5Barg-type%5D%0A%09%09%09%09%09%09if%20isinstance%28cfg_shared%5Bk%5D%2C%20Callable%29%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09%09%09%09else%20f%22%7Bk%7D%3D%7Bcfg_shared%5Bk%5D%7D%22%0A%60%60%60&labels=TYPING)

  ```python
  [
  	f"{k}={cfg_shared[k].__name__}"
  	# TYPING: error: Argument 2 to "isinstance" has incompatible type "<typing special form>"; expected "_ClassInfo"  [arg-type]
  	if isinstance(cfg_shared[k], Callable)  # type: ignore[arg-type]
  	else f"{k}={cfg_shared[k]}"
  ```


- error: Argument "param_values" to "analyze" of "SweepResult" has incompatible type "float | list[float] | list[list[float]] | list[list[list[Any]]]"; expected "list[Any]"  [arg-type]  
  local link: [`/maze_dataset/benchmark/config_sweep.py:428`](/maze_dataset/benchmark/config_sweep.py#L428) 
  | view on GitHub: [maze_dataset/benchmark/config_sweep.py#L428](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/config_sweep.py#L428)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%20%22param_values%22%20to%20%22analyze%22%20of%20%22SweepResult%22%20has%20incompatible%20type%20%22float%20%7C%20list%5Bfloat%5D%20%7C%20list%5Blist%5Bfloat%5D%5D%20%7C%20list%5Blist%5Blist%5BAny%5D%5D%5D%22%3B%20expected%20%22list%5BAny%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L428%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fconfig_sweep.py%23L428%29%0A%0A%23%20context%0A%60%60%60python%0A%09result%3A%20SweepResult%20%3D%20SweepResult.analyze%28%0A%09%09configs%3Dconfigs%2C%20%20%23%20type%3A%20ignore%5Bmisc%5D%0A%09%09%23%20TYPING%3A%20error%3A%20Argument%20%22param_values%22%20to%20%22analyze%22%20of%20%22SweepResult%22%20has%20incompatible%20type%20%22float%20%7C%20list%5Bfloat%5D%20%7C%20list%5Blist%5Bfloat%5D%5D%20%7C%20list%5Blist%5Blist%5BAny%5D%5D%5D%22%3B%20expected%20%22list%5BAny%5D%22%20%20%5Barg-type%5D%0A%09%09param_values%3Dnp.linspace%280.0%2C%201.0%2C%20p_val_count%29.tolist%28%29%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09param_key%3D%22maze_ctor_kwargs.p%22%2C%0A%60%60%60&labels=TYPING)

  ```python
  result: SweepResult = SweepResult.analyze(
  	configs=configs,  # type: ignore[misc]
  	# TYPING: error: Argument "param_values" to "analyze" of "SweepResult" has incompatible type "float | list[float] | list[list[float]] | list[list[list[Any]]]"; expected "list[Any]"  [arg-type]
  	param_values=np.linspace(0.0, 1.0, p_val_count).tolist(),  # type: ignore[arg-type]
  	param_key="maze_ctor_kwargs.p",
  ```




## [`maze_dataset/benchmark/sweep_fit.py`](/maze_dataset/benchmark/sweep_fit.py)

- error: Incompatible types in assignment (expression has type "floating[Any]", variable has type "float")  [assignment]  
  local link: [`/maze_dataset/benchmark/sweep_fit.py:334`](/maze_dataset/benchmark/sweep_fit.py#L334) 
  | view on GitHub: [maze_dataset/benchmark/sweep_fit.py#L334](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/benchmark/sweep_fit.py#L334)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Incompatible%20types%20in%20assignment%20%28expression%20has%20type%20%22floating%5BAny%5D%22%2C%20variable%20has%20type%20%22float%22%29%20%20%5Bassignment%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fbenchmark%2Fsweep_fit.py%23L334%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fbenchmark%2Fsweep_fit.py%23L334%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%23%20Calculate%20f%28x%2Cp%29%20for%20all%20combinations%0A%09%09%09for%20i%2C%20p_val%20in%20enumerate%28ps%29%3A%0A%09%09%09%09%23%20TYPING%3A%20error%3A%20Incompatible%20types%20in%20assignment%20%28expression%20has%20type%20%22floating%5BAny%5D%22%2C%20variable%20has%20type%20%22float%22%29%20%20%5Bassignment%5D%0A%09%09%09%09for%20j%2C%20x_val%20in%20enumerate%28xs%29%3A%20%20%23%20type%3A%20ignore%5Bassignment%5D%0A%09%09%09%09%09Z%5Bi%2C%20j%5D%20%3D%20soft_step%28x_val%2C%20p_val%2C%20alpha%2C%20w%29%0A%60%60%60&labels=TYPING)

  ```python
  # Calculate f(x,p) for all combinations
  for i, p_val in enumerate(ps):
  	# TYPING: error: Incompatible types in assignment (expression has type "floating[Any]", variable has type "float")  [assignment]
  	for j, x_val in enumerate(xs):  # type: ignore[assignment]
  		Z[i, j] = soft_step(x_val, p_val, alpha, w)
  ```




## [`maze_dataset/dataset/configs.py`](/maze_dataset/dataset/configs.py)

- error: Return type "list[str]" of "keys" incompatible with return type "KeysView[str]" in supertype "Mapping"  [override]  
  local link: [`/maze_dataset/dataset/configs.py:58`](/maze_dataset/dataset/configs.py#L58) 
  | view on GitHub: [maze_dataset/dataset/configs.py#L58](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/configs.py#L58)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Return%20type%20%22list%5Bstr%5D%22%20of%20%22keys%22%20incompatible%20with%20return%20type%20%22KeysView%5Bstr%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fconfigs.py%23L58%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fconfigs.py%23L58%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20iter%28self._configs%29%0A%0A%09%23%20TYPING%3A%20error%3A%20Return%20type%20%22list%5Bstr%5D%22%20of%20%22keys%22%20incompatible%20with%20return%20type%20%22KeysView%5Bstr%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D%0A%09def%20keys%28self%29%20-%3E%20list%5Bstr%5D%3A%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%09%09%22return%20the%20keys%22%0A%60%60%60&labels=TYPING)

  ```python
  	return iter(self._configs)

  # TYPING: error: Return type "list[str]" of "keys" incompatible with return type "KeysView[str]" in supertype "Mapping"  [override]
  def keys(self) -> list[str]:  # type: ignore[override]
  	"return the keys"
  ```


- error: Return type "list[tuple[str, MazeDatasetConfig]]" of "items" incompatible with return type "ItemsView[str, MazeDatasetConfig]" in supertype "Mapping"  [override]  
  local link: [`/maze_dataset/dataset/configs.py:63`](/maze_dataset/dataset/configs.py#L63) 
  | view on GitHub: [maze_dataset/dataset/configs.py#L63](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/configs.py#L63)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Return%20type%20%22list%5Btuple%5Bstr%2C%20MazeDatasetConfig%5D%5D%22%20of%20%22items%22%20incompatible%20with%20return%20type%20%22ItemsView%5Bstr%2C%20MazeDatasetConfig%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fconfigs.py%23L63%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fconfigs.py%23L63%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20list%28self._configs.keys%28%29%29%0A%0A%09%23%20TYPING%3A%20error%3A%20Return%20type%20%22list%5Btuple%5Bstr%2C%20MazeDatasetConfig%5D%5D%22%20of%20%22items%22%20incompatible%20with%20return%20type%20%22ItemsView%5Bstr%2C%20MazeDatasetConfig%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D%0A%09def%20items%28self%29%20-%3E%20list%5Btuple%5Bstr%2C%20MazeDatasetConfig%5D%5D%3A%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%09%09%22return%20the%20items%22%0A%60%60%60&labels=TYPING)

  ```python
  	return list(self._configs.keys())

  # TYPING: error: Return type "list[tuple[str, MazeDatasetConfig]]" of "items" incompatible with return type "ItemsView[str, MazeDatasetConfig]" in supertype "Mapping"  [override]
  def items(self) -> list[tuple[str, MazeDatasetConfig]]:  # type: ignore[override]
  	"return the items"
  ```


- error: Return type "list[MazeDatasetConfig]" of "values" incompatible with return type "ValuesView[MazeDatasetConfig]" in supertype "Mapping"  [override]  
  local link: [`/maze_dataset/dataset/configs.py:68`](/maze_dataset/dataset/configs.py#L68) 
  | view on GitHub: [maze_dataset/dataset/configs.py#L68](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/configs.py#L68)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Return%20type%20%22list%5BMazeDatasetConfig%5D%22%20of%20%22values%22%20incompatible%20with%20return%20type%20%22ValuesView%5BMazeDatasetConfig%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fconfigs.py%23L68%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fconfigs.py%23L68%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20%5B%28k%2C%20copy.deepcopy%28v%29%29%20for%20k%2C%20v%20in%20self._configs.items%28%29%5D%0A%0A%09%23%20TYPING%3A%20error%3A%20Return%20type%20%22list%5BMazeDatasetConfig%5D%22%20of%20%22values%22%20incompatible%20with%20return%20type%20%22ValuesView%5BMazeDatasetConfig%5D%22%20in%20supertype%20%22Mapping%22%20%20%5Boverride%5D%0A%09def%20values%28self%29%20-%3E%20list%5BMazeDatasetConfig%5D%3A%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%09%09return%20%5Bcopy.deepcopy%28v%29%20for%20v%20in%20self._configs.values%28%29%5D%0A%60%60%60&labels=TYPING)

  ```python
  	return [(k, copy.deepcopy(v)) for k, v in self._configs.items()]

  # TYPING: error: Return type "list[MazeDatasetConfig]" of "values" incompatible with return type "ValuesView[MazeDatasetConfig]" in supertype "Mapping"  [override]
  def values(self) -> list[MazeDatasetConfig]:  # type: ignore[override]
  	return [copy.deepcopy(v) for v in self._configs.values()]
  ```




## [`maze_dataset/dataset/dataset.py`](/maze_dataset/dataset/dataset.py)

- error: Argument 1 to "len" has incompatible type "GPTDatasetConfig"; expected "Sized"  [arg-type]  
  local link: [`/maze_dataset/dataset/dataset.py:125`](/maze_dataset/dataset/dataset.py#L125) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L125](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L125)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20to%20%22len%22%20has%20incompatible%20type%20%22GPTDatasetConfig%22%3B%20expected%20%22Sized%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L125%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L125%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%29%0A%09%09return%20sanitize_fname%28%0A%09%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22len%22%20has%20incompatible%20type%20%22GPTDatasetConfig%22%3B%20expected%20%22Sized%22%20%20%5Barg-type%5D%0A%09%09%09f%22f%7Bself.name%7D-n%7Bshorten_numerical_to_str%28len%28self%29%29%7D-h%7Bself_json_hash%7D%22%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%29%0A%60%60%60&labels=TYPING)

  ```python
  )
  return sanitize_fname(
  	# TYPING: error: Argument 1 to "len" has incompatible type "GPTDatasetConfig"; expected "Sized"  [arg-type]
  	f"f{self.name}-n{shorten_numerical_to_str(len(self))}-h{self_json_hash}",  # type: ignore[arg-type]
  )
  ```


- error: ParamSpec "P_FilterKwargs" is unbound  [valid-type]  
  local link: [`/maze_dataset/dataset/dataset.py:539`](/maze_dataset/dataset/dataset.py#L539) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L539](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L539)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20ParamSpec%20%22P_FilterKwargs%22%20is%20unbound%20%20%5Bvalid-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L539%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L539%29%0A%0A%23%20context%0A%60%60%60python%0A%09%40functools.wraps%28method%29%0A%09def%20wrapper%28%0A%09%09%23%20TYPING%3A%20error%3A%20ParamSpec%20%22P_FilterKwargs%22%20is%20unbound%20%20%5Bvalid-type%5D%0A%09%09dataset%3A%20T_Dataset%2C%0A%09%09%2Aargs%3A%20P_FilterKwargs.args%2C%20%20%23%20type%3A%20ignore%5Bvalid-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  @functools.wraps(method)
  def wrapper(
  	# TYPING: error: ParamSpec "P_FilterKwargs" is unbound  [valid-type]
  	dataset: T_Dataset,
  	*args: P_FilterKwargs.args,  # type: ignore[valid-type]
  ```


- error: Incompatible return value type (got "_Wrapped[[Any, KwArg(Any)], Any, [Never, VarArg(Any), KwArg(Any)], Never]", expected "DatasetFilterProtocol[Any]")  [return-value]  
  local link: [`/maze_dataset/dataset/dataset.py:552`](/maze_dataset/dataset/dataset.py#L552) 
  | view on GitHub: [maze_dataset/dataset/dataset.py#L552](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/dataset.py#L552)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Incompatible%20return%20value%20type%20%28got%20%22_Wrapped%5B%5BAny%2C%20KwArg%28Any%29%5D%2C%20Any%2C%20%5BNever%2C%20VarArg%28Any%29%2C%20KwArg%28Any%29%5D%2C%20Never%5D%22%2C%20expected%20%22DatasetFilterProtocol%5BAny%5D%22%29%20%20%5Breturn-value%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fdataset.py%23L552%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fdataset.py%23L552%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20new_dataset%0A%0A%09%23%20TYPING%3A%20error%3A%20Incompatible%20return%20value%20type%20%28got%20%22_Wrapped%5B%5BAny%2C%20KwArg%28Any%29%5D%2C%20Any%2C%20%5BNever%2C%20VarArg%28Any%29%2C%20KwArg%28Any%29%5D%2C%20Never%5D%22%2C%20expected%20%22DatasetFilterProtocol%5BAny%5D%22%29%20%20%5Breturn-value%5D%0A%09return%20wrapper%20%20%23%20type%3A%20ignore%5Breturn-value%5D%0A%60%60%60&labels=TYPING)

  ```python
  	return new_dataset

  # TYPING: error: Incompatible return value type (got "_Wrapped[[Any, KwArg(Any)], Any, [Never, VarArg(Any), KwArg(Any)], Never]", expected "DatasetFilterProtocol[Any]")  [return-value]
  return wrapper  # type: ignore[return-value]
  ```




## [`maze_dataset/dataset/maze_dataset.py`](/maze_dataset/dataset/maze_dataset.py)

- error: No overload variant of "generate_random_path" of "LatticeMaze" matches argument type "dict[Literal['allowed_start', 'allowed_end', 'deadend_start', 'deadend_end', 'endpoints_not_equal', 'except_on_no_valid_endpoint'], bool | list[tuple[int, int]] | None]"  [call-overload]  
  local link: [`/maze_dataset/dataset/maze_dataset.py:61`](/maze_dataset/dataset/maze_dataset.py#L61) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L61](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L61)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20No%20overload%20variant%20of%20%22generate_random_path%22%20of%20%22LatticeMaze%22%20matches%20argument%20type%20%22dict%5BLiteral%5B%27allowed_start%27%2C%20%27allowed_end%27%2C%20%27deadend_start%27%2C%20%27deadend_end%27%2C%20%27endpoints_not_equal%27%2C%20%27except_on_no_valid_endpoint%27%5D%2C%20bool%20%7C%20list%5Btuple%5Bint%2C%20int%5D%5D%20%7C%20None%5D%22%20%20%5Bcall-overload%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L61%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L61%29%0A%0A%23%20context%0A%60%60%60python%0A%09%23%20Generate%20the%20solution%0A%09%23%20mypy%20doesnt%20realize%20EndpointKwargsType%20has%20only%20string%20keys%3A%20%60Keywords%20must%20be%20strings%20%20%5Bmisc%5D%60%0A%09%23%20TYPING%3A%20error%3A%20No%20overload%20variant%20of%20%22generate_random_path%22%20of%20%22LatticeMaze%22%20matches%20argument%20type%20%22dict%5BLiteral%5B%27allowed_start%27%2C%20%27allowed_end%27%2C%20%27deadend_start%27%2C%20%27deadend_end%27%2C%20%27endpoints_not_equal%27%2C%20%27except_on_no_valid_endpoint%27%5D%2C%20bool%20%7C%20list%5Btuple%5Bint%2C%20int%5D%5D%20%7C%20None%5D%22%20%20%5Bcall-overload%5D%0A%09solution%3A%20Optional%5BCoordArray%5D%20%3D%20maze.generate_random_path%28%2A%2Aendpoint_kwargs%29%20%20%23%20type%3A%20ignore%5Bmisc%2C%20call-overload%5D%0A%60%60%60&labels=TYPING)

  ```python
  # Generate the solution
  # mypy doesnt realize EndpointKwargsType has only string keys: `Keywords must be strings  [misc]`
  # TYPING: error: No overload variant of "generate_random_path" of "LatticeMaze" matches argument type "dict[Literal['allowed_start', 'allowed_end', 'deadend_start', 'deadend_end', 'endpoints_not_equal', 'except_on_no_valid_endpoint'], bool | list[tuple[int, int]] | None]"  [call-overload]
  solution: Optional[CoordArray] = maze.generate_random_path(**endpoint_kwargs)  # type: ignore[misc, call-overload]
  ```


- error: Return type "MazeDataset" of "from_config" incompatible with return type "T_Dataset" in supertype "GPTDataset"  [override]  
  local link: [`/maze_dataset/dataset/maze_dataset.py:127`](/maze_dataset/dataset/maze_dataset.py#L127) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L127](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L127)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Return%20type%20%22MazeDataset%22%20of%20%22from_config%22%20incompatible%20with%20return%20type%20%22T_Dataset%22%20in%20supertype%20%22GPTDataset%22%20%20%5Boverride%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L127%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L127%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09self.generation_metadata_collected%3A%20dict%20%7C%20None%20%3D%20generation_metadata_collected%0A%0A%09%23%20TYPING%3A%20error%3A%20Return%20type%20%22MazeDataset%22%20of%20%22from_config%22%20incompatible%20with%20return%20type%20%22T_Dataset%22%20in%20supertype%20%22GPTDataset%22%20%20%5Boverride%5D%0A%09%40classmethod%0A%09def%20from_config%28%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%60%60%60&labels=TYPING)

  ```python
  	self.generation_metadata_collected: dict | None = generation_metadata_collected

  # TYPING: error: Return type "MazeDataset" of "from_config" incompatible with return type "T_Dataset" in supertype "GPTDataset"  [override]
  @classmethod
  def from_config(  # type: ignore[override]
  ```


- error: Argument 1 of "from_config" is incompatible with supertype "GPTDataset"; supertype defines the argument type as "T_DatasetConfig"  [override]  
  local link: [`/maze_dataset/dataset/maze_dataset.py:131`](/maze_dataset/dataset/maze_dataset.py#L131) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L131](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L131)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20of%20%22from_config%22%20is%20incompatible%20with%20supertype%20%22GPTDataset%22%3B%20supertype%20defines%20the%20argument%20type%20as%20%22T_DatasetConfig%22%20%20%5Boverride%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L131%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L131%29%0A%0A%23%20context%0A%60%60%60python%0A%09def%20from_config%28%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%09%09cls%2C%0A%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20of%20%22from_config%22%20is%20incompatible%20with%20supertype%20%22GPTDataset%22%3B%20supertype%20defines%20the%20argument%20type%20as%20%22T_DatasetConfig%22%20%20%5Boverride%5D%0A%09%09cfg%3A%20MazeDatasetConfig%2C%20%20%23%20type%3A%20ignore%5Boverride%5D%0A%09%09do_generate%3A%20bool%20%3D%20True%2C%0A%60%60%60&labels=TYPING)

  ```python
  def from_config(  # type: ignore[override]
  	cls,
  	# TYPING: error: Argument 1 of "from_config" is incompatible with supertype "GPTDataset"; supertype defines the argument type as "T_DatasetConfig"  [override]
  	cfg: MazeDatasetConfig,  # type: ignore[override]
  	do_generate: bool = True,
  ```


- get type hints on the tokenizer here  
  local link: [`/maze_dataset/dataset/maze_dataset.py:188`](/maze_dataset/dataset/maze_dataset.py#L188) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L188](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L188)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=get%20type%20hints%20on%20the%20tokenizer%20here&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L188%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L188%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20MazeDataset.load%28self._serialize_full%28%29%29%0A%0A%09%23%20TYPING%3A%20get%20type%20hints%20on%20the%20tokenizer%20here%0A%09%40overload%0A%09def%20as_tokens%28%0A%60%60%60&labels=TYPING)

  ```python
  	return MazeDataset.load(self._serialize_full())

  # TYPING: get type hints on the tokenizer here
  @overload
  def as_tokens(
  ```


- error: Argument 1 to "map" has incompatible type "Callable[[int], SolvedMaze | None]"; expected "Callable[[str], SolvedMaze | None]"  [arg-type]  
  local link: [`/maze_dataset/dataset/maze_dataset.py:296`](/maze_dataset/dataset/maze_dataset.py#L296) 
  | view on GitHub: [maze_dataset/dataset/maze_dataset.py#L296](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/maze_dataset.py#L296)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20to%20%22map%22%20has%20incompatible%20type%20%22Callable%5B%5Bint%5D%2C%20SolvedMaze%20%7C%20None%5D%22%3B%20expected%20%22Callable%5B%5Bstr%5D%2C%20SolvedMaze%20%7C%20None%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fmaze_dataset.py%23L296%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fmaze_dataset.py%23L296%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09%09tqdm.tqdm%28%0A%09%09%09%09%09map%28%0A%09%09%09%09%09%09%23%20TYPING%3A%20%20error%3A%20Argument%201%20to%20%22map%22%20has%20incompatible%20type%20%22Callable%5B%5Bint%5D%2C%20SolvedMaze%20%7C%20None%5D%22%3B%20expected%20%22Callable%5B%5Bstr%5D%2C%20SolvedMaze%20%7C%20None%5D%22%20%20%5Barg-type%5D%0A%09%09%09%09%09%09%23%20why%20does%20it%20think%20tolist%28%29%20returns%20a%20string%3F%0A%09%09%09%09%09%09_generate_maze_helper%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  tqdm.tqdm(
  	map(
  		# TYPING:  error: Argument 1 to "map" has incompatible type "Callable[[int], SolvedMaze | None]"; expected "Callable[[str], SolvedMaze | None]"  [arg-type]
  		# why does it think tolist() returns a string?
  		_generate_maze_helper,  # type: ignore[arg-type]
  ```




## [`maze_dataset/dataset/rasterized.py`](/maze_dataset/dataset/rasterized.py)

- error: Attributes without a default cannot follow attributes with one  [misc]  
  local link: [`/maze_dataset/dataset/rasterized.py:120`](/maze_dataset/dataset/rasterized.py#L120) 
  | view on GitHub: [maze_dataset/dataset/rasterized.py#L120](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/rasterized.py#L120)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Attributes%20without%20a%20default%20cannot%20follow%20attributes%20with%20one%20%20%5Bmisc%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Frasterized.py%23L120%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Frasterized.py%23L120%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TYPING%3A%20error%3A%20Attributes%20without%20a%20default%20cannot%20follow%20attributes%20with%20one%20%20%5Bmisc%5D%0A%40serializable_dataclass%0Aclass%20RasterizedMazeDatasetConfig%28MazeDatasetConfig%29%3A%20%20%23%20type%3A%20ignore%5Bmisc%5D%0A%60%60%60&labels=TYPING)

  ```python
  # TYPING: error: Attributes without a default cannot follow attributes with one  [misc]
  @serializable_dataclass
  class RasterizedMazeDatasetConfig(MazeDatasetConfig):  # type: ignore[misc]
  ```




## [`maze_dataset/dataset/success_predict_math.py`](/maze_dataset/dataset/success_predict_math.py)

- this is messed up, some of these args can be arrays but i dont remember which?  
  local link: [`/maze_dataset/dataset/success_predict_math.py:68`](/maze_dataset/dataset/success_predict_math.py#L68) 
  | view on GitHub: [maze_dataset/dataset/success_predict_math.py#L68](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/dataset/success_predict_math.py#L68)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20messed%20up%2C%20some%20of%20these%20args%20can%20be%20arrays%20but%20i%20dont%20remember%20which%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Fdataset%2Fsuccess_predict_math.py%23L68%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fdataset%2Fsuccess_predict_math.py%23L68%29%0A%0A%23%20context%0A%60%60%60python%0A%09https%3A%2F%2Fwww.desmos.com%2Fcalculator%2Fqllvhwftvy%0A%09%22%22%22%0A%09%23%20TYPING%3A%20this%20is%20messed%20up%2C%20some%20of%20these%20args%20can%20be%20arrays%20but%20i%20dont%20remember%20which%3F%0A%09return%20h_func%28%0A%09%09x%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  https://www.desmos.com/calculator/qllvhwftvy
  """
  # TYPING: this is messed up, some of these args can be arrays but i dont remember which?
  return h_func(
  	x,  # type: ignore[arg-type]
  ```




## [`maze_dataset/generation/generators.py`](/maze_dataset/generation/generators.py)

- error: Dict entry 1 has incompatible type  
  local link: [`/maze_dataset/generation/generators.py:615`](/maze_dataset/generation/generators.py#L615) 
  | view on GitHub: [maze_dataset/generation/generators.py#L615](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/generation/generators.py#L615)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Dict%20entry%201%20has%20incompatible%20type&body=%23%20source%0A%0A%5B%60maze_dataset%2Fgeneration%2Fgenerators.py%23L615%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fgeneration%2Fgenerators.py%23L615%29%0A%0A%23%20context%0A%60%60%60python%0AGENERATORS_MAP%3A%20dict%5Bstr%2C%20Callable%5B%5BCoord%20%7C%20CoordTup%2C%20Any%5D%2C%20%22LatticeMaze%22%5D%5D%20%3D%20%7B%0A%09%22gen_dfs%22%3A%20LatticeMazeGenerators.gen_dfs%2C%0A%09%23%20TYPING%3A%20error%3A%20Dict%20entry%201%20has%20incompatible%20type%0A%09%23%20%22str%22%3A%20%22Callable%5B%5Bndarray%5BAny%2C%20Any%5D%20%7C%20tuple%5Bint%2C%20int%5D%2C%20KwArg%28Any%29%5D%2C%20LatticeMaze%5D%22%3B%0A%09%23%20expected%20%22str%22%3A%20%22Callable%5B%5Bndarray%5BAny%2C%20Any%5D%20%7C%20tuple%5Bint%2C%20int%5D%2C%20Any%5D%2C%20LatticeMaze%5D%22%20%20%5Bdict-item%5D%0A%60%60%60&labels=TYPING)

  ```python
  GENERATORS_MAP: dict[str, Callable[[Coord | CoordTup, Any], "LatticeMaze"]] = {
  	"gen_dfs": LatticeMazeGenerators.gen_dfs,
  	# TYPING: error: Dict entry 1 has incompatible type
  	# "str": "Callable[[ndarray[Any, Any] | tuple[int, int], KwArg(Any)], LatticeMaze]";
  	# expected "str": "Callable[[ndarray[Any, Any] | tuple[int, int], Any], LatticeMaze]"  [dict-item]
  ```


- error: Too few arguments  [call-arg]  
  local link: [`/maze_dataset/generation/generators.py:647`](/maze_dataset/generation/generators.py#L647) 
  | view on GitHub: [maze_dataset/generation/generators.py#L647](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/generation/generators.py#L647)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Too%20few%20arguments%20%20%5Bcall-arg%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Fgeneration%2Fgenerators.py%23L647%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fgeneration%2Fgenerators.py%23L647%29%0A%0A%23%20context%0A%60%60%60python%0A%09if%20maze_ctor_kwargs%20is%20None%3A%0A%09%09maze_ctor_kwargs%20%3D%20dict%28%29%0A%09%23%20TYPING%3A%20error%3A%20Too%20few%20arguments%20%20%5Bcall-arg%5D%0A%09%23%20not%20sure%20why%20this%20is%20happening%20--%20doesnt%20recognize%20the%20kwargs%3F%0A%09maze%3A%20LatticeMaze%20%3D%20GENERATORS_MAP%5Bgen_name%5D%28grid_shape%2C%20%2A%2Amaze_ctor_kwargs%29%20%20%23%20type%3A%20ignore%5Bcall-arg%5D%0A%60%60%60&labels=TYPING)

  ```python
  if maze_ctor_kwargs is None:
  	maze_ctor_kwargs = dict()
  # TYPING: error: Too few arguments  [call-arg]
  # not sure why this is happening -- doesnt recognize the kwargs?
  maze: LatticeMaze = GENERATORS_MAP[gen_name](grid_shape, **maze_ctor_kwargs)  # type: ignore[call-arg]
  ```




## [`maze_dataset/plotting/print_tokens.py`](/maze_dataset/plotting/print_tokens.py)

- this is missing a lot of type hints  
  local link: [`/maze_dataset/plotting/print_tokens.py:90`](/maze_dataset/plotting/print_tokens.py#L90) 
  | view on GitHub: [maze_dataset/plotting/print_tokens.py#L90](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/print_tokens.py#L90)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=this%20is%20missing%20a%20lot%20of%20type%20hints&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fprint_tokens.py%23L90%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fprint_tokens.py%23L90%29%0A%0A%23%20context%0A%60%60%60python%0A%09if%20max_length%20is%20not%20None%3A%0A%09%09%23%20TODO%3A%20why%20are%20we%20using%20a%20map%20here%20again%3F%0A%09%09%23%20TYPING%3A%20this%20is%20missing%20a%20lot%20of%20type%20hints%0A%09%09wrapped%3A%20list%20%3D%20list%28%20%20%23%20noqa%3A%20C417%0A%09%09%09map%28%0A%60%60%60&labels=TYPING)

  ```python
  if max_length is not None:
  	# TODO: why are we using a map here again?
  	# TYPING: this is missing a lot of type hints
  	wrapped: list = list(  # noqa: C417
  		map(
  ```


- would be nice to type hint as html, latex, or terminal string and overload depending on `FormatType`  
  local link: [`/maze_dataset/plotting/print_tokens.py:123`](/maze_dataset/plotting/print_tokens.py#L123) 
  | view on GitHub: [maze_dataset/plotting/print_tokens.py#L123](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/plotting/print_tokens.py#L123)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=would%20be%20nice%20to%20type%20hint%20as%20html%2C%20latex%2C%20or%20terminal%20string%20and%20overload%20depending%20on%20%60FormatType%60&body=%23%20source%0A%0A%5B%60maze_dataset%2Fplotting%2Fprint_tokens.py%23L123%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Fplotting%2Fprint_tokens.py%23L123%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TYPING%3A%20would%20be%20nice%20to%20type%20hint%20as%20html%2C%20latex%2C%20or%20terminal%20string%20and%20overload%20depending%20on%20%60FormatType%60%0Adef%20color_tokens_cmap%28%0A%09tokens%3A%20list%5Bstr%5D%2C%0A%60%60%60&labels=TYPING)

  ```python
  # TYPING: would be nice to type hint as html, latex, or terminal string and overload depending on `FormatType`
  def color_tokens_cmap(
  	tokens: list[str],
  ```




## [`maze_dataset/tokenization/modular/all_instances.py`](/maze_dataset/tokenization/modular/all_instances.py)

- Incompatible return value type (got "filter[FiniteValued]", expected "Generator[FiniteValued, None, None]")  [return-value]  
  local link: [`/maze_dataset/tokenization/modular/all_instances.py:98`](/maze_dataset/tokenization/modular/all_instances.py#L98) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_instances.py#L98](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_instances.py#L98)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=Incompatible%20return%20value%20type%20%28got%20%22filter%5BFiniteValued%5D%22%2C%20expected%20%22Generator%5BFiniteValued%2C%20None%2C%20None%5D%22%29%20%20%5Breturn-value%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L98%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L98%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20vals%0A%09if%20type_%20in%20validation_funcs%3A%20%20%23%20Only%20possible%20catch%20of%20UnionTypes%0A%09%09%23%20TYPING%3A%20Incompatible%20return%20value%20type%20%28got%20%22filter%5BFiniteValued%5D%22%2C%20expected%20%22Generator%5BFiniteValued%2C%20None%2C%20None%5D%22%29%20%20%5Breturn-value%5D%0A%09%09return%20filter%28validation_funcs%5Btype_%5D%2C%20vals%29%0A%09elif%20hasattr%28%0A%60%60%60&labels=TYPING)

  ```python
  	return vals
  if type_ in validation_funcs:  # Only possible catch of UnionTypes
  	# TYPING: Incompatible return value type (got "filter[FiniteValued]", expected "Generator[FiniteValued, None, None]")  [return-value]
  	return filter(validation_funcs[type_], vals)
  elif hasattr(
  ```


- error: Incompatible types in assignment (expression has type "filter[FiniteValued]", variable has type "Generator[FiniteValued, None, None]")  [assignment]  
  local link: [`/maze_dataset/tokenization/modular/all_instances.py:107`](/maze_dataset/tokenization/modular/all_instances.py#L107) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_instances.py#L107](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_instances.py#L107)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Incompatible%20types%20in%20assignment%20%28expression%20has%20type%20%22filter%5BFiniteValued%5D%22%2C%20variable%20has%20type%20%22Generator%5BFiniteValued%2C%20None%2C%20None%5D%22%29%20%20%5Bassignment%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L107%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L107%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09if%20superclass%20not%20in%20validation_funcs%3A%0A%09%09%09%09continue%0A%09%09%09%23%20TYPING%3A%20error%3A%20Incompatible%20types%20in%20assignment%20%28expression%20has%20type%20%22filter%5BFiniteValued%5D%22%2C%20variable%20has%20type%20%22Generator%5BFiniteValued%2C%20None%2C%20None%5D%22%29%20%20%5Bassignment%5D%0A%09%09%09vals%20%3D%20filter%28validation_funcs%5Bsuperclass%5D%2C%20vals%29%0A%09%09%09break%20%20%23%20Only%20the%20first%20validation%20function%20hit%20in%20the%20mro%20is%20applied%0A%60%60%60&labels=TYPING)

  ```python
  if superclass not in validation_funcs:
  	continue
  # TYPING: error: Incompatible types in assignment (expression has type "filter[FiniteValued]", variable has type "Generator[FiniteValued, None, None]")  [assignment]
  vals = filter(validation_funcs[superclass], vals)
  break  # Only the first validation function hit in the mro is applied
  ```


- some better type hints would be nice here  
  local link: [`/maze_dataset/tokenization/modular/all_instances.py:121`](/maze_dataset/tokenization/modular/all_instances.py#L121) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_instances.py#L121](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_instances.py#L121)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=some%20better%20type%20hints%20would%20be%20nice%20here&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L121%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_instances.py%23L121%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TYPING%3A%20some%20better%20type%20hints%20would%20be%20nice%20here%0Adef%20_all_instances_wrapper%28f%3A%20Callable%29%20-%3E%20Callable%3A%0A%09%22%22%22Converts%20dicts%20to%20frozendicts%20to%20allow%20caching%20and%20applies%20%60_apply_validation_func%60.%22%22%22%0A%60%60%60&labels=TYPING)

  ```python
  # TYPING: some better type hints would be nice here
  def _all_instances_wrapper(f: Callable) -> Callable:
  	"""Converts dicts to frozendicts to allow caching and applies `_apply_validation_func`."""
  ```




## [`maze_dataset/tokenization/modular/all_tokenizers.py`](/maze_dataset/tokenization/modular/all_tokenizers.py)

- error: Type variable "maze_dataset.utils.FiniteValued" is unbound  [valid-type]  
  local link: [`/maze_dataset/tokenization/modular/all_tokenizers.py:52`](/maze_dataset/tokenization/modular/all_tokenizers.py#L52) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_tokenizers.py#L52](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_tokenizers.py#L52)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Type%20variable%20%22maze_dataset.utils.FiniteValued%22%20is%20unbound%20%20%5Bvalid-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L52%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L52%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20Always%20include%20this%20as%20the%20first%20item%20in%20the%20dict%20%60validation_funcs%60%20whenever%20using%20%60all_instances%60%20with%20%60MazeTokenizerModular%60%0A%23%20TYPING%3A%20error%3A%20Type%20variable%20%22maze_dataset.utils.FiniteValued%22%20is%20unbound%20%20%5Bvalid-type%5D%0A%23%20%20%20note%3A%20%28Hint%3A%20Use%20%22Generic%5BFiniteValued%5D%22%20or%20%22Protocol%5BFiniteValued%5D%22%20base%20class%20to%20bind%20%22FiniteValued%22%20inside%20a%20class%29%0A%23%20%20%20note%3A%20%28Hint%3A%20Use%20%22FiniteValued%22%20in%20function%20signature%20to%20bind%20%22FiniteValued%22%20inside%20a%20function%29%0A%60%60%60&labels=TYPING)

  ```python
  # Always include this as the first item in the dict `validation_funcs` whenever using `all_instances` with `MazeTokenizerModular`
  # TYPING: error: Type variable "maze_dataset.utils.FiniteValued" is unbound  [valid-type]
  #   note: (Hint: Use "Generic[FiniteValued]" or "Protocol[FiniteValued]" base class to bind "FiniteValued" inside a class)
  #   note: (Hint: Use "FiniteValued" in function signature to bind "FiniteValued" inside a function)
  ```


- Item "bool" of the upper bound "bool | IsDataclass | Enum" of type variable "FiniteValued" has no attribute "is_valid"  [union-attr]  
  local link: [`/maze_dataset/tokenization/modular/all_tokenizers.py:60`](/maze_dataset/tokenization/modular/all_tokenizers.py#L60) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_tokenizers.py#L60](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_tokenizers.py#L60)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=Item%20%22bool%22%20of%20the%20upper%20bound%20%22bool%20%7C%20IsDataclass%20%7C%20Enum%22%20of%20type%20variable%20%22FiniteValued%22%20has%20no%20attribute%20%22is_valid%22%20%20%5Bunion-attr%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L60%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L60%29%0A%0A%23%20context%0A%60%60%60python%0A%5D%20%3D%20frozendict.frozendict%28%0A%09%7B%0A%09%09%23%20TYPING%3A%20Item%20%22bool%22%20of%20the%20upper%20bound%20%22bool%20%7C%20IsDataclass%20%7C%20Enum%22%20of%20type%20variable%20%22FiniteValued%22%20has%20no%20attribute%20%22is_valid%22%20%20%5Bunion-attr%5D%0A%09%09_TokenizerElement%3A%20lambda%20x%3A%20x.is_valid%28%29%2C%0A%09%09%23%20Currently%20no%20need%20for%20%60MazeTokenizerModular.is_valid%60%20since%20that%20method%20contains%20no%20special%20cases%20not%20already%20covered%20by%20%60_TokenizerElement.is_valid%60%0A%60%60%60&labels=TYPING)

  ```python
  ] = frozendict.frozendict(
  	{
  		# TYPING: Item "bool" of the upper bound "bool | IsDataclass | Enum" of type variable "FiniteValued" has no attribute "is_valid"  [union-attr]
  		_TokenizerElement: lambda x: x.is_valid(),
  		# Currently no need for `MazeTokenizerModular.is_valid` since that method contains no special cases not already covered by `_TokenizerElement.is_valid`
  ```


- error: No overload variant of "set" matches argument type "FiniteValued"  [call-overload]  
  local link: [`/maze_dataset/tokenization/modular/all_tokenizers.py:64`](/maze_dataset/tokenization/modular/all_tokenizers.py#L64) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_tokenizers.py#L64](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_tokenizers.py#L64)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20No%20overload%20variant%20of%20%22set%22%20matches%20argument%20type%20%22FiniteValued%22%20%20%5Bcall-overload%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L64%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L64%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20Currently%20no%20need%20for%20%60MazeTokenizerModular.is_valid%60%20since%20that%20method%20contains%20no%20special%20cases%20not%20already%20covered%20by%20%60_TokenizerElement.is_valid%60%0A%09%09%23%20MazeTokenizerModular%3A%20lambda%20x%3A%20x.is_valid%28%29%2C%0A%09%09%23%20TYPING%3A%20error%3A%20No%20overload%20variant%20of%20%22set%22%20matches%20argument%20type%20%22FiniteValued%22%20%20%5Bcall-overload%5D%0A%09%09%23%20%20%20note%3A%20Possible%20overload%20variants%3A%0A%09%09%23%20%20%20note%3A%20%20%20%20%20def%20%5B_T%5D%20set%28self%29%20-%3E%20set%5B_T%5D%0A%60%60%60&labels=TYPING)

  ```python
  # Currently no need for `MazeTokenizerModular.is_valid` since that method contains no special cases not already covered by `_TokenizerElement.is_valid`
  # MazeTokenizerModular: lambda x: x.is_valid(),
  # TYPING: error: No overload variant of "set" matches argument type "FiniteValued"  [call-overload]
  #   note: Possible overload variants:
  #   note:     def [_T] set(self) -> set[_T]
  ```


- error: Argument 1 to "len" has incompatible type "FiniteValued"; expected "Sized"  [arg-type]  
  local link: [`/maze_dataset/tokenization/modular/all_tokenizers.py:68`](/maze_dataset/tokenization/modular/all_tokenizers.py#L68) 
  | view on GitHub: [maze_dataset/tokenization/modular/all_tokenizers.py#L68](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/all_tokenizers.py#L68)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20to%20%22len%22%20has%20incompatible%20type%20%22FiniteValued%22%3B%20expected%20%22Sized%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L68%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Fall_tokenizers.py%23L68%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%23%20%20%20note%3A%20%20%20%20%20def%20%5B_T%5D%20set%28self%29%20-%3E%20set%5B_T%5D%0A%09%09%23%20%20%20note%3A%20%20%20%20%20def%20%5B_T%5D%20set%28self%2C%20Iterable%5B_T%5D%2C%20%2F%29%20-%3E%20set%5B_T%5D%0A%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22len%22%20has%20incompatible%20type%20%22FiniteValued%22%3B%20expected%20%22Sized%22%20%20%5Barg-type%5D%0A%09%09StepTokenizers.StepTokenizerPermutation%3A%20lambda%20x%3A%20len%28set%28x%29%29%20%3D%3D%20len%28x%29%0A%09%09and%20x%20%21%3D%20%28StepTokenizers.Distance%28%29%2C%29%2C%0A%60%60%60&labels=TYPING)

  ```python
  #   note:     def [_T] set(self) -> set[_T]
  #   note:     def [_T] set(self, Iterable[_T], /) -> set[_T]
  # TYPING: error: Argument 1 to "len" has incompatible type "FiniteValued"; expected "Sized"  [arg-type]
  StepTokenizers.StepTokenizerPermutation: lambda x: len(set(x)) == len(x)
  and x != (StepTokenizers.Distance(),),
  ```




## [`maze_dataset/tokenization/modular/element_base.py`](/maze_dataset/tokenization/modular/element_base.py)

- type hint `v` more specifically  
  local link: [`/maze_dataset/tokenization/modular/element_base.py:43`](/maze_dataset/tokenization/modular/element_base.py#L43) 
  | view on GitHub: [maze_dataset/tokenization/modular/element_base.py#L43](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/element_base.py#L43)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=type%20hint%20%60v%60%20more%20specifically&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L43%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L43%29%0A%0A%23%20context%0A%60%60%60python%0A%09%22%22%22%0A%0A%09%23%20TYPING%3A%20type%20hint%20%60v%60%20more%20specifically%0A%09%40staticmethod%0A%09def%20_stringify%28k%3A%20str%2C%20v%3A%20Any%29%20-%3E%20str%3A%20%20%23%20noqa%3A%20ANN401%0A%60%60%60&labels=TYPING)

  ```python
  """

  # TYPING: type hint `v` more specifically
  @staticmethod
  def _stringify(k: str, v: Any) -> str:  # noqa: ANN401
  ```


- type hints for `__init_subclass__`?  
  local link: [`/maze_dataset/tokenization/modular/element_base.py:69`](/maze_dataset/tokenization/modular/element_base.py#L69) 
  | view on GitHub: [maze_dataset/tokenization/modular/element_base.py#L69](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/element_base.py#L69)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=type%20hints%20for%20%60__init_subclass__%60%3F&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L69%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L69%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09return%20self.name%0A%0A%09%23%20TYPING%3A%20type%20hints%20for%20%60__init_subclass__%60%3F%0A%09def%20__init_subclass__%28cls%2C%20%2A%2Akwargs%29%3A%20%20%23%20noqa%3A%20ANN204%0A%09%09%22%22%22Hack%3A%20dataclass%20hashes%20don%27t%20include%20the%20class%20itself%20in%20the%20hash%20function%20inputs.%0A%60%60%60&labels=TYPING)

  ```python
  	return self.name

  # TYPING: type hints for `__init_subclass__`?
  def __init_subclass__(cls, **kwargs):  # noqa: ANN204
  	"""Hack: dataclass hashes don't include the class itself in the hash function inputs.
  ```


- better type hints for this function  
  local link: [`/maze_dataset/tokenization/modular/element_base.py:268`](/maze_dataset/tokenization/modular/element_base.py#L268) 
  | view on GitHub: [maze_dataset/tokenization/modular/element_base.py#L268](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/element_base.py#L268)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=better%20type%20hints%20for%20this%20function&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L268%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Felement_base.py%23L268%29%0A%0A%23%20context%0A%60%60%60python%0A%23%20TYPING%3A%20better%20type%20hints%20for%20this%20function%0Adef%20mark_as_unsupported%28is_valid%3A%20Callable%5B%5BT%2C%20bool%5D%2C%20bool%5D%29%20-%3E%20Callable%5B%5BT%5D%2C%20T%5D%3A%0A%09%22%22%22mark%20a%20_TokenizerElement%20as%20unsupported.%0A%60%60%60&labels=TYPING)

  ```python
  # TYPING: better type hints for this function
  def mark_as_unsupported(is_valid: Callable[[T, bool], bool]) -> Callable[[T], T]:
  	"""mark a _TokenizerElement as unsupported.
  ```




## [`maze_dataset/tokenization/modular/fst.py`](/maze_dataset/tokenization/modular/fst.py)

- add a protocol or abc for both of these which is a context manager that takes the args we care about  
  local link: [`/maze_dataset/tokenization/modular/fst.py:37`](/maze_dataset/tokenization/modular/fst.py#L37) 
  | view on GitHub: [maze_dataset/tokenization/modular/fst.py#L37](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/tokenization/modular/fst.py#L37)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=add%20a%20protocol%20or%20abc%20for%20both%20of%20these%20which%20is%20a%20context%20manager%20that%20takes%20the%20args%20we%20care%20about&body=%23%20source%0A%0A%5B%60maze_dataset%2Ftokenization%2Fmodular%2Ffst.py%23L37%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Ftokenization%2Fmodular%2Ffst.py%23L37%29%0A%0A%23%20context%0A%60%60%60python%0A%29%20-%3E%20FstSet%3A%0A%09%22%22%22get%20all%20the%20tokenizers%2C%20save%20an%20fst%20file%20at%20%60MMT_FST_PATH%60%20and%20return%20the%20set%22%22%22%0A%09%23%20TYPING%3A%20add%20a%20protocol%20or%20abc%20for%20both%20of%20these%20which%20is%20a%20context%20manager%20that%20takes%20the%20args%20we%20care%20about%0A%09%23%20probably%20do%20this%20in%20muutils%0A%09sp%3A%20type%5BSpinnerContext%20%7C%20NoOpContextManager%5D%20%3D%20%28%0A%60%60%60&labels=TYPING)

  ```python
  ) -> FstSet:
  	"""get all the tokenizers, save an fst file at `MMT_FST_PATH` and return the set"""
  	# TYPING: add a protocol or abc for both of these which is a context manager that takes the args we care about
  	# probably do this in muutils
  	sp: type[SpinnerContext | NoOpContextManager] = (
  ```




## [`maze_dataset/utils.py`](/maze_dataset/utils.py)

- error: Overloaded function signature 2 will never be matched: signature 1's parameter type(s) are the same or broader  [overload-cannot-match]  
  local link: [`/maze_dataset/utils.py:95`](/maze_dataset/utils.py#L95) 
  | view on GitHub: [maze_dataset/utils.py#L95](https://github.com/understanding-search/maze-dataset/blob/main/maze_dataset/utils.py#L95)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Overloaded%20function%20signature%202%20will%20never%20be%20matched%3A%20signature%201%27s%20parameter%20type%28s%29%20are%20the%20same%20or%20broader%20%20%5Boverload-cannot-match%5D&body=%23%20source%0A%0A%5B%60maze_dataset%2Futils.py%23L95%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Fmaze_dataset%2Futils.py%23L95%29%0A%0A%23%20context%0A%60%60%60python%0A%09edges%3A%20Int%5Bnp.ndarray%2C%20%22edges%20coord%3D2%20row_col%3D2%22%5D%2C%0A%29%20-%3E%20Int8%5Bnp.ndarray%2C%20%22%20edges%22%5D%3A%20...%0A%23%20TYPING%3A%20error%3A%20Overloaded%20function%20signature%202%20will%20never%20be%20matched%3A%20signature%201%27s%20parameter%20type%28s%29%20are%20the%20same%20or%20broader%20%20%5Boverload-cannot-match%5D%0A%23%20this%20is%20because%20mypy%20doesn%27t%20play%20nice%20with%20jaxtyping%0A%40overload%0A%60%60%60&labels=TYPING)

  ```python
  	edges: Int[np.ndarray, "edges coord=2 row_col=2"],
  ) -> Int8[np.ndarray, " edges"]: ...
  # TYPING: error: Overloaded function signature 2 will never be matched: signature 1's parameter type(s) are the same or broader  [overload-cannot-match]
  # this is because mypy doesn't play nice with jaxtyping
  @overload
  ```




## [`tests/unit/tokenization/test_all_instances.py`](/tests/unit/tokenization/test_all_instances.py)

- error: Argument 2 to "dataclass_set_equals" has incompatible type "Iterable[FiniteValued]"; expected "Iterable[IsDataclass]"  [arg-type]  
  local link: [`/tests/unit/tokenization/test_all_instances.py:289`](/tests/unit/tokenization/test_all_instances.py#L289) 
  | view on GitHub: [tests/unit/tokenization/test_all_instances.py#L289](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/tokenization/test_all_instances.py#L289)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%202%20to%20%22dataclass_set_equals%22%20has%20incompatible%20type%20%22Iterable%5BFiniteValued%5D%22%3B%20expected%20%22Iterable%5BIsDataclass%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60tests%2Funit%2Ftokenization%2Ftest_all_instances.py%23L289%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Ftokenization%2Ftest_all_instances.py%23L289%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09%09list%28all_instances%28type_%2C%20validation_funcs%29%29%0A%09elif%20hasattr%28type_%2C%20%22__dataclass_fields__%22%29%3A%0A%09%09%23%20TYPING%3A%20error%3A%20Argument%202%20to%20%22dataclass_set_equals%22%20has%20incompatible%20type%20%22Iterable%5BFiniteValued%5D%22%3B%20expected%20%22Iterable%5BIsDataclass%5D%22%20%20%5Barg-type%5D%0A%09%09assert%20dataclass_set_equals%28all_instances%28type_%2C%20validation_funcs%29%2C%20result%29%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09else%3A%20%20%23%20General%20case%2C%20due%20to%20nesting%2C%20results%20might%20contain%20some%20dataclasses%20and%20some%20other%20types%0A%60%60%60&labels=TYPING)

  ```python
  		list(all_instances(type_, validation_funcs))
  elif hasattr(type_, "__dataclass_fields__"):
  	# TYPING: error: Argument 2 to "dataclass_set_equals" has incompatible type "Iterable[FiniteValued]"; expected "Iterable[IsDataclass]"  [arg-type]
  	assert dataclass_set_equals(all_instances(type_, validation_funcs), result)  # type: ignore[arg-type]
  else:  # General case, due to nesting, results might contain some dataclasses and some other types
  ```


- error: Argument 1 to "filter" has incompatible type "Callable[[Any], bool]"; expected "Callable[[FiniteValued], TypeGuard[IsDataclass]]"  [arg-type]  
  local link: [`/tests/unit/tokenization/test_all_instances.py:294`](/tests/unit/tokenization/test_all_instances.py#L294) 
  | view on GitHub: [tests/unit/tokenization/test_all_instances.py#L294](https://github.com/understanding-search/maze-dataset/blob/main/tests/unit/tokenization/test_all_instances.py#L294)
  | [Make Issue](https://github.com/understanding-search/maze-dataset/issues/new?title=error%3A%20Argument%201%20to%20%22filter%22%20has%20incompatible%20type%20%22Callable%5B%5BAny%5D%2C%20bool%5D%22%3B%20expected%20%22Callable%5B%5BFiniteValued%5D%2C%20TypeGuard%5BIsDataclass%5D%5D%22%20%20%5Barg-type%5D&body=%23%20source%0A%0A%5B%60tests%2Funit%2Ftokenization%2Ftest_all_instances.py%23L294%60%5D%28https%3A%2F%2Fgithub.com%2Funderstanding-search%2Fmaze-dataset%2Fblob%2Fmain%2Ftests%2Funit%2Ftokenization%2Ftest_all_instances.py%23L294%29%0A%0A%23%20context%0A%60%60%60python%0A%09%09out%20%3D%20list%28all_instances%28type_%2C%20validation_funcs%29%29%0A%09%09assert%20dataclass_set_equals%28%0A%09%09%09%23%20TYPING%3A%20error%3A%20Argument%201%20to%20%22filter%22%20has%20incompatible%20type%20%22Callable%5B%5BAny%5D%2C%20bool%5D%22%3B%20expected%20%22Callable%5B%5BFiniteValued%5D%2C%20TypeGuard%5BIsDataclass%5D%5D%22%20%20%5Barg-type%5D%0A%09%09%09filter%28lambda%20x%3A%20isinstance%28x%2C%20IsDataclass%29%2C%20out%29%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%09%09%09filter%28lambda%20x%3A%20isinstance%28x%2C%20IsDataclass%29%2C%20result%29%2C%20%20%23%20type%3A%20ignore%5Barg-type%5D%0A%60%60%60&labels=TYPING)

  ```python
  out = list(all_instances(type_, validation_funcs))
  assert dataclass_set_equals(
  	# TYPING: error: Argument 1 to "filter" has incompatible type "Callable[[Any], bool]"; expected "Callable[[FiniteValued], TypeGuard[IsDataclass]]"  [arg-type]
  	filter(lambda x: isinstance(x, IsDataclass), out),  # type: ignore[arg-type]
  	filter(lambda x: isinstance(x, IsDataclass), result),  # type: ignore[arg-type]
  ```




