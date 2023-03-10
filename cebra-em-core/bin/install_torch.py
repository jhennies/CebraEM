
import sys

if __name__ == '__main__':

	# Platform-specific imports
	if sys.platform == 'linux':
		# Linux specific imports here
		from cebra_em_core.linux.cuda import custom_pytorch

	elif sys.platform[:3] == 'win':
		# Windows specific imports here
		from cebra_em_core.windows.cuda import custom_pytorch
	else:
		raise NotImplementedError(f'Workflow not implemented for {sys.platform}')

	custom_pytorch()
