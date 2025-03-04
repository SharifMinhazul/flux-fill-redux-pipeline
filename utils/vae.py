import utils.misc.utility
import utils.misc.sd

def load_vae(vae_path: str = "models/vae/ae.safetensors"):
        sd = utils.misc.utility.load_torch_file(vae_path)
        return utils.misc.sd.VAE(sd=sd)