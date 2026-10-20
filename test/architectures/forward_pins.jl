# The outputs of the architectures that are built from the SympNet, attention, volume-preserving
# and PSD layers, against values recorded on commit 4c4a8320 with Julia 1.13.2 on arm64. The
# networks, their seeds and their input are in `test/helpers/architecture_pins.jl`.
#
# A recorded value is compared at 8 ULPs of the larger of the two magnitudes, not with `==`: a value
# that passes through GEMM and `tanh` differs by 1 to 2 ULPs between x86_64 and arm64, while a
# wrong layer, a swapped `q` and `p` branch or a wrong parameter slice moves it by a fraction.

using GeometricMachineLearning
using GeometricMachineLearning: BiasLayer, UnknownEncoder, UnknownDecoder,
                                UnknownSymplecticEncoder, UnknownSymplecticDecoder,
                                NonLinearSymplecticEncoder, NonLinearSymplecticDecoder,
                                AutoEncoder, params
using Test

include("../helpers/eltypes.jl")
include("../helpers/architecture_pins.jl")

const ULPS = 8

function within_ulps(a, b)
    all(abs(x - y) ≤ ULPS * eps(max(abs(x), abs(y))) for (x, y) in zip(a, b))
end

const RECORDED = Dict(
    ("GSympNet init_upper", Float32) => Float32[
        1.2079871f0, 1.700258f0, 0.6440241f0, 0.31081647f0, -1.111841f0, -2.0382395f0,
        -0.9300919f0, -0.5819578f0, 0.8181355f0, 2.046548f0, 1.110849f0, 0.7946684f0],
    ("GSympNet init_upper", Float64) =>
        Float64[2.8616602457474687, 1.1647107270915928, 1.5917873324276202,
            0.6820691347879498, -2.667503267384504, -1.0933032451713,
            -1.7271093243367208, -0.9335804621281338, 2.2808470497543762,
            0.9650697923225338, 1.6748430011793989, 1.0827190230128103],
    ("GSympNet init_lower", Float32) =>
        Float32[1.3259686f0, 1.1507093f0, 0.8284635f0, 0.32455733f0,
            -1.257479f0, -1.1678908f0, -1.0798897f0, -0.61549103f0,
            1.045658f0, 1.0486478f0, 1.2166196f0, 0.8316883f0],
    ("GSympNet init_lower", Float64) =>
        Float64[0.9459997821193615, 1.489669601528201, 1.5529384032913474,
            1.2916850317542126, -0.7413855301502071, -1.5499030644425937,
            -1.781017910287798, -1.5574487683103349, 0.4496624231030051,
            1.4487769069845253, 1.8459871086947133, 1.708797851802824],
    ("LASympNet true true", Float32) =>
        Float32[0.91563416f0, 8.735449f0, -11.838316f0, -10.375892f0,
            -0.32519674f0, -3.7658088f0, 2.9511719f0, 5.848351f0,
            1.170372f0, 5.0407534f0, -10.12165f0, -6.3045955f0],
    ("LASympNet true true", Float64) =>
        Float64[-0.35967535206058815, 0.15850972212388248, 0.6979061630831969,
            -2.252725601762963, -2.983256448121466, -4.1242346022841305,
            15.32263796151871, 15.629232132091806, 1.5613956192329563,
            1.8413211892402748, -8.147646688182828, -9.527117314482345],
    ("LASympNet true false", Float32) =>
        Float32[2.978579f0, 6.34558f0, -8.887037f0, -7.4144683f0, -3.4995546f0, 1.085736f0,
            5.3186584f0, 1.8480694f0, 3.0591981f0, 2.9335968f0, -6.729628f0, -3.3890796f0],
    ("LASympNet true false", Float64) =>
        Float64[1.5875955398647947, -1.5312414626145592, -3.3676166806028363,
            0.9490128263868745, -2.3495102762750415, -4.253774979883703,
            11.439515919826057, 15.073950245615295, 2.4100474109347823,
            -0.5483398011086553, -8.85496142637767, -2.587296244626805],
    ("LASympNet false true", Float32) =>
        Float32[0.20517719f0, 3.507387f0, 1.1528658f0, -1.4951923f0,
            5.9821825f0, 1.9414268f0, -4.0281096f0, 2.1002264f0,
            -1.2210033f0, 1.4358667f0, 1.2483217f0, 0.054223478f0],
    ("LASympNet false true", Float64) =>
        Float64[-5.968919062936291, -16.747865495162394, 1.4344612961578673,
            3.866239961636913, 12.48076817679028, 6.5130196752436875,
            -4.853617451570946, 0.09715566495493244, -4.972846023173946,
            -15.590551249877606, 1.6671274000793432, 3.055446611758987],
    ("LASympNet false false", Float32) =>
        Float32[-5.5990915f0, 2.358982f0, 0.9907367f0, -2.146427f0,
            3.599338f0, 3.928326f0, -1.2205318f0, -1.9136994f0,
            -6.4082584f0, -0.47260118f0, 0.30867422f0, 0.2842155f0],
    ("LASympNet false false", Float64) =>
        Float64[-11.721475811538122, -17.57703583735072, 2.617563637982298,
            3.839466503997929, 19.754504161250814, 8.597932408585145,
            -6.303207244046076, -0.42319028820192217, -11.146855941835629,
            -17.75083116801041, 3.053336235891592, 3.2540451099335126],
    ("LinearSymplecticTransformer init_upper", Float32) => Float32[
        1.0473452f0, 1.290153f0, 0.9862076f0, 0.12439126f0, -0.371012f0, -0.433926f0,
        0.20295194f0, 1.4495585f0, 1.5350857f0, 1.37661f0, 2.1724093f0, 1.1424887f0],
    ("LinearSymplecticTransformer init_upper", Float64) =>
        Float64[1.8142211442799514, 1.3263271500570728, 1.80441956225232,
            0.9132320833006399, -1.7050381214042678, -1.9354302918926403,
            -2.0853097059101375, -0.8599420740729693, 0.9080088820365366,
            1.4653363232383732, 1.150535065738142, 0.8690544454163456],
    ("LinearSymplecticTransformer init_lower", Float32) =>
        Float32[1.2002544f0, 1.472335f0, 0.43544662f0, -1.0060117f0,
            -0.82764184f0, -0.99508816f0, -2.0758927f0, 0.52672005f0,
            0.96206576f0, 0.45997345f0, 0.6908124f0, -0.30416498f0],
    ("LinearSymplecticTransformer init_lower", Float64) =>
        Float64[0.3453286578072775, 1.0309382375474154, -0.3188259553658408,
            0.9579650709120607, 1.0997251266830783, -1.0665817109450795,
            -0.10870563307620831, -2.823503897989008, -0.1412186096006332,
            1.1097308978589675, -0.1492869391284491, 2.273502654746609],
    ("SymplecticTransformer symmetric", Float32) =>
        Float32[0.66011816f0, 0.4031512f0, 0.77980375f0, 0.7759167f0,
            -0.15857296f0, -0.39368695f0, -0.74829334f0, -0.13491812f0,
            0.6793265f0, 0.93753743f0, 2.7067904f0, 2.9560735f0],
    ("SymplecticTransformer symmetric", Float64) =>
        Float64[1.712116678777355, 0.48927907797410214, 1.4440525613591952,
            -0.2578727526343351, -1.4139990065820451, -1.013268147112556,
            -1.4981628929201867, -0.23268565443982436, 1.2000031923882333,
            0.6012318414564659, 1.5546739500637636, 0.32630132282366103],
    ("SymplecticTransformer arbitrary", Float32) =>
        Float32[1.1091877f0, 0.58851945f0, 2.8974078f0, 2.5699472f0,
            -0.8892055f0, -0.71547455f0, -1.9215745f0, -1.7115502f0,
            -0.15935874f0, 1.3219812f0, 1.4522033f0, 2.9305377f0],
    ("SymplecticTransformer arbitrary", Float64) =>
        Float64[-1.1368781887525148, 1.030621379256654, 2.379156212686098,
            0.9979001322473816, 0.9338289764247059, -0.9838074905013856,
            -0.7488494571507978, 0.02200028562015549, -1.556186432712347,
            0.7520771517124243, 1.0634370794648507, 0.34137672275751285],
    ("SymplecticTransformer upscaled", Float32) =>
        Float32[1.2091758f0, 0.07274055f0, -0.8966182f0, 0.24525768f0,
            -1.5140926f0, -0.2895341f0, 0.6047826f0, -0.06663975f0,
            0.6532773f0, 0.36584884f0, 1.3078529f0, 0.79046667f0],
    ("SymplecticTransformer upscaled", Float64) =>
        Float64[0.8909220472705279, 0.5009653747723455, 0.4707179961049061,
            -0.2151707909619296, -1.739987440332722, 0.8690565332367634,
            -2.9923881864648716, 2.1335704506240116, 1.1420288865817798,
            0.0730919768868471, 1.102510851216419, -0.31514784561797726],
    ("VolumePreservingFeedForward init_lower", Float32) =>
        Float32[-2.6498818f0, 2.6475134f0, -2.2106795f0, 1.2256302f0, -0.22352493f0,
            0.69771063f0, 1.8773587f0, -1.8569232f0, 1.9363346f0],
    ("VolumePreservingFeedForward init_lower", Float64) =>
        Float64[-1.9730335112100201, 2.096623285704716, 0.9397728343274279,
            1.1196246519833908, 0.5836313852580939, -0.9718630022903947,
            -3.023108937185734, 0.4381187800617896, 2.192409526040456],
    ("VolumePreservingFeedForward init_upper", Float32) =>
        Float32[-0.31070602f0, 1.2645521f0, 1.378274f0, -0.8653427f0,
            -3.0852795f0, -2.590437f0, 1.5881162f0, 2.127181f0, 0.9577617f0],
    ("VolumePreservingFeedForward init_upper", Float64) =>
        Float64[-0.8492501405906726, 1.2178758632519906, -0.9853113843660246,
            1.354578636940429, -2.3304567307510085, 2.2545352867528705,
            -0.3021880592609541, 1.0556286257060754, -1.203560177773379],
    ("VolumePreservingTransformer skew_sym", Float32) =>
        Float32[2.2717226f0, -0.80610025f0, 0.9336589f0, -2.208637f0, 0.61160004f0,
            -2.038826f0, 0.0014086068f0, 0.26194197f0, 1.3932756f0],
    ("VolumePreservingTransformer skew_sym", Float64) =>
        Float64[1.3225773020116778, -0.5237881146450325, 1.459439556845456,
            1.9984739907060318, -0.47141475156596435, 2.0015098769790445,
            -3.117757587368102, 2.022761649212967, -3.8284028623691073],
    ("VolumePreservingTransformer arbitrary", Float32) =>
        Float32[1.3559175f0, -0.3834728f0, 1.2342659f0, -2.2168589f0, 2.4734697f0,
            -3.2415924f0, 0.46869206f0, -2.2776783f0, 1.8075072f0],
    ("VolumePreservingTransformer arbitrary", Float64) =>
        Float64[-1.123109966596294, 1.2008825402260785, 0.08229843583416219,
            2.4235802532406847, -5.522066635989703, 3.2308400674181668,
            0.6758315991041173, -0.5409226981629305, -0.11710341490590837],
    ("StandardTransformerIntegrator", Float32) =>
        Float32[2.7102492f0, 0.8409711f0, 1.7109538f0, 1.4613752f0,
            -6.633284f0, -2.601374f0, -2.1084824f0, -1.2327838f0,
            1.6070602f0, 0.40005243f0, 1.3909433f0, 1.490759f0],
    ("StandardTransformerIntegrator", Float64) =>
        Float64[1.6765795376471715, 2.0253826828371224, 1.9396444792767356,
            -0.4065594425284016, 0.9580921946462989, 2.390290742854175,
            -1.2099841313524893, -0.6295540869944397, 0.6249683186048016,
            0.30638051859371285, 1.2862625487003583, 0.34771896969300675],
    ("StandardTransformerIntegrator upscaled", Float32) =>
        Float32[5.0004373f0, -3.3843598f0, 4.0245004f0, -5.777192f0,
            5.456158f0, -1.9904926f0, 5.592142f0, -7.0980053f0,
            5.5051756f0, -1.9602716f0, 4.3036985f0, -5.1615086f0],
    ("StandardTransformerIntegrator upscaled", Float64) =>
        Float64[5.781867460012214, -1.8877286879344843, 5.561493238527312,
            -0.4512903517147898, 5.325904126259073, -2.305216634988211,
            3.2885540831052253, -1.98950005700121, 4.698104540986305,
            -3.7047507436500977, 4.0419882581616875, -3.232224408988411],
    ("Transformer Stiefel", Float32) =>
        Float32[0.6554814f0, -1.0019424f0, -0.5829763f0, 0.26682833f0,
            -0.1362085f0, 0.22772393f0, 0.10464845f0, -0.06153867f0,
            0.6136032f0, -1.106835f0, -0.54371595f0, 0.12815577f0],
    ("Transformer Stiefel", Float64) =>
        Float64[0.7488146807464766, -0.26836762005085824, -1.2009358049211174,
            0.8477293053889665, -0.16613672161294007, 0.07059383013342774,
            0.2361632700442593, -0.1870022811137076, 0.8394936127710724,
            -0.3718117005173244, -1.0870117839045188, 0.7853648924331973],
    ("SymplecticAutoencoder true true whole", Float32) =>
        Float32[0.034231946f0, -0.9673753f0, 0.0f0, -1.1813196f0, 0.03519869f0,
            1.2740035f0, 0.0f0, 1.4145536f0, 0.02958343f0, -0.769261f0, 0.0f0,
            -1.4387347f0, 0.21450384f0, 1.2676231f0, 0.0f0, 1.1481509f0],
    ("SymplecticAutoencoder true true whole", Float64) =>
        Float64[-0.2995595631979327, -1.8619911815271186, 0.0, -1.349522533236108,
            0.16485534648829514, 1.4523572257063484, 0.0, 0.8937068108159045,
            -0.26060131705657474, -1.6367565948369442, 0.0, -1.5382663742127787,
            0.17270784568792874, 1.012504358068519, 0.0, 0.9481202993983309],
    ("SymplecticAutoencoder true true encoder", Float32) =>
        Float32[1.0396041f0, -0.6824867f0, -0.85159284f0, 1.1586751f0,
            0.85417086f0, -0.9233718f0, -0.64491576f0, 1.0675846f0],
    ("SymplecticAutoencoder true true encoder", Float64) =>
        Float64[1.1545113857891114, -0.7062079096861748, -1.1767564300839901,
            0.9408143873729808, 1.0302787320730635, -0.9652049072014992,
            -0.777716309146846, 0.9793235830618591],
    ("SymplecticAutoencoder true true decoder", Float32) =>
        Float32[-0.04574212f0, -0.679162f0, 0.0f0, 0.81877583f0, -0.59496737f0,
            -1.2665386f0, 0.0f0, 0.55693746f0, 0.062503345f0, 0.36175683f0,
            0.0f0, -0.5733446f0, 0.38774604f0, 1.3187432f0, 0.0f0, -0.7293168f0],
    ("SymplecticAutoencoder true true decoder", Float64) =>
        Float64[0.10595072725541503, -0.5764870141005125, 0.0, 1.2340624951690038,
            -0.024928362750852077, -0.5558386462288921, 0.0, -0.006185663404346675,
            -0.097453819262312, 0.35942141461821553, 0.0, -1.1320406628688087,
            0.03322740211321147, 0.788068661492363, 0.0, -0.31066332990231277],
    ("SymplecticAutoencoder true false whole", Float32) =>
        Float32[-0.109970115f0, -1.569913f0, 0.0f0, -0.87776184f0, -0.04690169f0,
            0.9559243f0, 0.0f0, 1.3596908f0, -0.27286628f0, -1.5179324f0, 0.0f0,
            -0.92979974f0, -0.038817406f0, 0.64065975f0, 0.0f0, 1.4537175f0],
    ("SymplecticAutoencoder true false whole", Float64) =>
        Float64[-0.1852304779036989, -1.4136854438201734, 0.0, -0.6678435294777991,
            0.2768059664354866, 1.8309794828220574, 0.0, 1.55050636123056,
            -0.15394419447309157, -1.2284426093042053, 0.0, -0.9536145220362764,
            0.2872765960473799, 1.4499967393652569, 0.0, 1.600519627861937],
    ("SymplecticAutoencoder true false encoder", Float32) =>
        Float32[1.0396041f0, -0.6824867f0, -0.85159284f0, 1.1586751f0,
            0.85417086f0, -0.9233718f0, -0.64491576f0, 1.0675846f0],
    ("SymplecticAutoencoder true false encoder", Float64) =>
        Float64[1.1545113857891114, -0.7062079096861748, -1.1767564300839901,
            0.9408143873729808, 1.0302787320730635, -0.9652049072014992,
            -0.777716309146846, 0.9793235830618591],
    ("SymplecticAutoencoder true false decoder", Float32) =>
        Float32[0.07290539f0, -0.5632652f0, 0.0f0, 1.1137031f0, -0.05409482f0,
            -0.4107324f0, 0.0f0, -0.67689645f0, 0.16877776f0, 0.62166315f0,
            0.0f0, -1.3111036f0, 0.048105713f0, 0.6313648f0, 0.0f0, 0.3177796f0],
    ("SymplecticAutoencoder true false decoder", Float64) =>
        Float64[0.05764065562086877, -0.6941649216291748, 0.0, 0.9438409900756717,
            -0.006403318385174888, -0.6238782107378746, 0.0, -0.17535249205322215,
            -0.037372422620812494, 0.414499872311911, 0.0, -0.9240535044725269,
            0.03671120897825355, 0.9261239780564474, 0.0, -0.12380434435640827],
    ("SymplecticAutoencoder false true whole", Float32) =>
        Float32[0.042635076f0, -0.97409785f0, 0.0f0, -1.20973f0, 0.044231545f0,
            1.4836725f0, 0.0f0, 1.1873991f0, 0.030500058f0, -0.8759587f0,
            0.0f0, -1.4262561f0, 0.16641834f0, 1.2388972f0, 0.0f0, 1.0899969f0],
    ("SymplecticAutoencoder false true whole", Float64) =>
        Float64[-0.2913333301161063, -1.8748650862839338, 0.0, -1.3136127423504425,
            0.16462044298699224, 1.3842761771510554, 0.0, 0.8856029393900701,
            -0.26424002365732985, -1.5759853123063488, 0.0, -1.5440828614451465,
            0.1670749077558988, 1.031257712660235, 0.0, 0.9598916702529003],
    ("SymplecticAutoencoder false true encoder", Float32) =>
        Float32[0.94332963f0, -0.87333214f0, -1.045148f0, 0.9395008f0,
            0.9268527f0, -0.97056067f0, -0.6516942f0, 0.9490281f0],
    ("SymplecticAutoencoder false true encoder", Float64) =>
        Float64[1.1852842498801963, -0.6856415267720114, -1.1054515814563868,
            0.9343434238136794, 0.9603346644194907, -0.9643338292602635,
            -0.8065601983412033, 0.986255229234491],
    ("SymplecticAutoencoder false true decoder", Float32) =>
        Float32[-0.04574212f0, -0.679162f0, 0.0f0, 0.81877583f0, -0.59496737f0,
            -1.2665386f0, 0.0f0, 0.55693746f0, 0.062503345f0, 0.36175683f0,
            0.0f0, -0.5733446f0, 0.38774604f0, 1.3187432f0, 0.0f0, -0.7293168f0],
    ("SymplecticAutoencoder false true decoder", Float64) =>
        Float64[0.10595072725541503, -0.5764870141005125, 0.0, 1.2340624951690038,
            -0.024928362750852077, -0.5558386462288921, 0.0, -0.006185663404346675,
            -0.097453819262312, 0.35942141461821553, 0.0, -1.1320406628688087,
            0.03322740211321147, 0.788068661492363, 0.0, -0.31066332990231277],
    ("SymplecticAutoencoder false false whole", Float32) =>
        Float32[-0.02638277f0, -1.3838918f0, 0.0f0, -1.1643248f0, -0.04002061f0,
            1.07502f0, 0.0f0, 1.270997f0, -0.22817582f0, -1.5290644f0, 0.0f0,
            -1.0053841f0, -0.03992005f0, 0.64795077f0, 0.0f0, 1.3377643f0],
    ("SymplecticAutoencoder false false whole", Float64) =>
        Float64[-0.17938511699247484, -1.4344632933867387, 0.0, -0.6519416156806348,
            0.2774647249319055, 1.7618882243128886, 0.0, 1.5471779299126251,
            -0.15630980043156364, -1.1611008182856941, 0.0, -0.9510317445367586,
            0.2794954802156819, 1.4602563831432092, 0.0, 1.5936843098196836],
    ("SymplecticAutoencoder false false encoder", Float32) =>
        Float32[0.94332963f0, -0.87333214f0, -1.045148f0, 0.9395008f0,
            0.9268527f0, -0.97056067f0, -0.6516942f0, 0.9490281f0],
    ("SymplecticAutoencoder false false encoder", Float64) =>
        Float64[1.1852842498801963, -0.6856415267720114, -1.1054515814563868,
            0.9343434238136794, 0.9603346644194907, -0.9643338292602635,
            -0.8065601983412033, 0.986255229234491],
    ("SymplecticAutoencoder false false decoder", Float32) =>
        Float32[0.07290539f0, -0.5632652f0, 0.0f0, 1.1137031f0, -0.05409482f0,
            -0.4107324f0, 0.0f0, -0.67689645f0, 0.16877776f0, 0.62166315f0,
            0.0f0, -1.3111036f0, 0.048105713f0, 0.6313648f0, 0.0f0, 0.3177796f0],
    ("SymplecticAutoencoder false false decoder", Float64) =>
        Float64[0.05764065562086877, -0.6941649216291748, 0.0, 0.9438409900756717,
            -0.006403318385174888, -0.6238782107378746, 0.0, -0.17535249205322215,
            -0.037372422620812494, 0.414499872311911, 0.0, -0.9240535044725269,
            0.03671120897825355, 0.9261239780564474, 0.0, -0.12380434435640827],
    ("PSDArch whole", Float32) =>
        Float32[0.0f0, 0.0f0, -0.9916648f0, -0.6754632f0, 0.0f0, 0.0f0,
            0.9775301f0, 0.8834547f0, 0.0f0, 0.0f0, -0.8504366f0,
            -0.98935825f0, 0.0f0, 0.0f0, 0.62507063f0, 0.9809362f0],
    ("PSDArch whole", Float64) =>
        Float64[0.0, 0.0, -0.9916648104524686, -0.6754631805511513, 0.0, 0.0,
            0.9775301176650969, 0.8834546557201536, 0.0, 0.0, -0.8504366206285644,
            -0.9893582466233819, 0.0, 0.0, 0.6250706488928821, 0.9809362300664916],
    ("PSDArch encoder", Float32) =>
        Float32[0.9916648f0, -0.6754632f0, -0.9775301f0, 0.8834547f0,
            0.8504366f0, -0.98935825f0, -0.62507063f0, 0.9809362f0],
    ("PSDArch encoder", Float64) =>
        Float64[0.9916648104524686, -0.6754631805511513, -0.9775301176650969,
            0.8834546557201536, 0.8504366206285644, -0.9893582466233819,
            -0.6250706488928821, 0.9809362300664916],
    ("PSDArch decoder", Float32) =>
        Float32[0.0f0, 0.0f0, -0.84147096f0, 0.9916648f0, 0.0f0, 0.0f0,
            -0.6754632f0, 0.041580662f0, 0.0f0, 0.0f0, 0.6118579f0,
            -0.9775301f0, 0.0f0, 0.0f0, 0.8834547f0, -0.37387666f0],
    ("PSDArch decoder", Float64) =>
        Float64[0.0, 0.0, -0.8414709848078965, 0.9916648104524686, 0.0, 0.0,
            -0.6754631805511513, 0.041580662433290935, 0.0, 0.0, 0.6118578909427189,
            -0.9775301176650969, 0.0, 0.0, 0.8834546557201536, -0.37387666483023685]
)

@testset "$name ($T) gives the recorded output" for (name, f) in PIN_CASES,
    T in REAL_ELTYPES

    y = f(T)
    expected = RECORDED[(name, T)]
    @test eltype(y) === T
    @test length(y) == length(expected)
    @test within_ulps(vec(y), expected)
end

# The layer sequence of `LASympNet` for each `(init_upper_linear, init_upper_act)`, recorded on
# 4c4a8320: `LQ`/`LP` a `LinearLayerQ`/`LinearLayerP`, `AQ`/`AP` an `ActivationLayerQ`/
# `ActivationLayerP`, `B` a `BiasLayer`, all of dimension 4 with `tanh`.
const LASYMPNET_LAYERS = Dict(
    (true, true) => "LQ LP LQ B AQ AP LQ LP LQ B AQ AP LQ LP LQ B",
    (true, false) => "LQ LP LQ B AP AQ LQ LP LQ B AP AQ LQ LP LQ B",
    (false, true) => "LP LQ LP B AQ AP LP LQ LP B AQ AP LP LQ LP B",
    (false, false) => "LP LQ LP B AP AQ LP LQ LP B AP AQ LP LQ LP B")

const LAYER_TYPES = Dict("LQ" => LinearLayerQ{4, 4}, "LP" => LinearLayerP{4, 4},
    "AQ" => ActivationLayerQ{4, 4, typeof(tanh)}, "AP" =>
        ActivationLayerP{4, 4, typeof(tanh)},
    "B" => BiasLayer{4, 4})

@testset "LASympNet init_upper_linear = $lin, init_upper_act = $act: the recorded layers" for lin in (
        true, false),
    act in (true, false)

    arch = LASympNet(
        4; depth = 3, nhidden = 2, init_upper_linear = lin, init_upper_act = act)
    expected = Tuple(LAYER_TYPES[c] for c in split(LASYMPNET_LAYERS[(lin, act)]))
    @test typeof.(Chain(arch).layers) === expected
end

# A plain autoencoder of this file's own, so that the generic `encoder` and `decoder` have a caller
# that is neither a `SymplecticAutoencoder` nor a `PSDArch`.
struct PlainAE <: AutoEncoder
    full_dim::Int
    reduced_dim::Int
    n_encoder_blocks::Int
    n_decoder_blocks::Int
end

function GeometricMachineLearning.encoder_layers_from_iteration(::PlainAE, it::AbstractVector{<:Integer})
    (Dense(
        it[1], it[2], tanh),)
end
function GeometricMachineLearning.decoder_layers_from_iteration(::PlainAE, it::AbstractVector{<:Integer})
    (Dense(
        it[1], it[2], tanh),)
end

@testset "encoder and decoder keep the architecture types of 4c4a8320 ($T)" for T in REAL_ELTYPES
    sae = NeuralNetwork(SymplecticAutoencoder(8, 4; n_encoder_layers = 2, n_decoder_layers = 2), CPU(), T)
    @test encoder(sae).architecture isa NonLinearSymplecticEncoder
    @test decoder(sae).architecture isa NonLinearSymplecticDecoder

    psd = NeuralNetwork(PSDArch(8, 4), CPU(), T)
    @test encoder(psd).architecture isa UnknownSymplecticEncoder
    @test decoder(psd).architecture isa UnknownSymplecticDecoder

    plain = NeuralNetwork(PlainAE(6, 2, 2, 2), CPU(), T)
    @test encoder(plain).architecture isa UnknownEncoder
    @test decoder(plain).architecture isa UnknownDecoder
    @test eltype(encoder(plain)(pin_input(T, 6, 2))) === T
    @test eltype(decoder(plain)(pin_input(T, 2, 2))) === T
end
